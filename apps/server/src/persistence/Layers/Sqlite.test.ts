import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { DatabaseSync } from "node:sqlite";

import * as NodeServices from "@effect/platform-node/NodeServices";
import { Effect, Layer } from "effect";
import * as SqlClient from "effect/unstable/sql/SqlClient";
import { afterEach, describe, expect, it } from "vitest";

import { makeSqlitePersistenceLive } from "./Sqlite.ts";
import { resolveSqliteMemoryBudget } from "../sqliteMemoryBudget.ts";

const tempDirectories: Array<string> = [];

async function makeDbPath(): Promise<string> {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "synara-sqlite-live-"));
  tempDirectories.push(directory);
  return path.join(directory, "state.sqlite");
}

async function createNormalWalSnapshot(dbPath: string): Promise<Buffer> {
  const seedPath = `${dbPath}.seed`;
  const seed = new DatabaseSync(seedPath);
  try {
    seed.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA wal_autocheckpoint = 0;
      CREATE TABLE recovery_probe(value TEXT NOT NULL);
    `);
    seed.prepare("INSERT INTO recovery_probe(value) VALUES (?)").run("survives-recovery");
    await Promise.all(
      ["", "-wal", "-shm"].map((suffix) =>
        fs.copyFile(`${seedPath}${suffix}`, `${dbPath}${suffix}`),
      ),
    );
    return fs.readFile(`${dbPath}-shm`);
  } finally {
    seed.close();
  }
}

afterEach(async () => {
  await Promise.all(
    tempDirectories.splice(0).map((directory) => fs.rm(directory, { recursive: true })),
  );
});

describe("SQLite persistence", () => {
  it("owns the live WAL exclusively without exposing a shared-memory sidecar", async () => {
    const dbPath = await makeDbPath();

    await Effect.runPromise(
      Effect.gen(function* () {
        const sql = yield* SqlClient.SqlClient;
        const [lockingMode] = yield* sql<{ readonly locking_mode: string }>`
          PRAGMA locking_mode;
        `;
        const [journalMode] = yield* sql<{ readonly journal_mode: string }>`
          PRAGMA journal_mode;
        `;

        expect(lockingMode?.locking_mode).toBe("exclusive");
        expect(journalMode?.journal_mode).toBe("wal");

        // Large-database tuning must actually take effect, not silently
        // no-op: a reverted or ignored cache_size reintroduces multi-minute
        // startups on multi-GB event logs. mmap_size may be capped by the
        // runtime build, so only require that mapping is enabled at all.
        const [cacheSize] = yield* sql<{ readonly cache_size: number }>`
          PRAGMA cache_size;
        `;
        const [mmapSize] = yield* sql<{ readonly mmap_size: number }>`
          PRAGMA mmap_size;
        `;
        expect(cacheSize?.cache_size).toBe(
          resolveSqliteMemoryBudget(os.totalmem()).cacheSizePragma,
        );
        expect(mmapSize?.mmap_size).toBeGreaterThan(0);

        yield* sql`CREATE TABLE ownership_probe(value TEXT NOT NULL)`;
        yield* sql`INSERT INTO ownership_probe(value) VALUES ('owned-by-synara')`;
        yield* Effect.promise(async () => {
          await expect(fs.stat(`${dbPath}-shm`)).rejects.toMatchObject({ code: "ENOENT" });
        });

        let external: DatabaseSync | undefined;
        expect(() => {
          external = new DatabaseSync(dbPath, { readOnly: true });
          external.prepare("SELECT value FROM ownership_probe").get();
        }).toThrow(/database is locked/i);
        external?.close();

        const rows = yield* sql<{ readonly value: string }>`
          SELECT value FROM ownership_probe
        `;
        expect(rows).toEqual([{ value: "owned-by-synara" }]);
        yield* Effect.promise(async () => {
          await expect(fs.stat(`${dbPath}-shm`)).rejects.toMatchObject({ code: "ENOENT" });
        });
      }).pipe(
        Effect.provide(makeSqlitePersistenceLive(dbPath).pipe(Layer.provide(NodeServices.layer))),
      ),
    );
  });

  it("recovers an existing WAL without touching its stale shared-memory sidecar", async () => {
    const dbPath = await makeDbPath();
    const staleShm = await createNormalWalSnapshot(dbPath);

    await Effect.runPromise(
      Effect.gen(function* () {
        const sql = yield* SqlClient.SqlClient;
        const rows = yield* sql<{ readonly value: string }>`
          SELECT value FROM recovery_probe
        `;
        const [lockingMode] = yield* sql<{ readonly locking_mode: string }>`
          PRAGMA locking_mode;
        `;

        expect(rows).toEqual([{ value: "survives-recovery" }]);
        expect(lockingMode?.locking_mode).toBe("exclusive");
        yield* Effect.promise(async () => {
          expect(await fs.readFile(`${dbPath}-shm`)).toEqual(staleShm);
        });
      }).pipe(
        Effect.provide(makeSqlitePersistenceLive(dbPath).pipe(Layer.provide(NodeServices.layer))),
      ),
    );

    expect(await fs.readFile(`${dbPath}-shm`)).toEqual(staleShm);
  });

  it("rebuilds a populated pre-auto-vacuum database once and drains its freelist", async () => {
    const dbPath = await makeDbPath();

    // A database written by <=1.0.7: churned rows deleted with auto_vacuum
    // still NONE, so every freed page is stranded on the freelist and the
    // file stays at its high-water mark.
    const legacy = new DatabaseSync(dbPath);
    try {
      legacy.exec("PRAGMA journal_mode = WAL");
      legacy.exec("CREATE TABLE churn(body TEXT NOT NULL)");
      const insert = legacy.prepare("INSERT INTO churn(body) VALUES (?)");
      for (let i = 0; i < 2000; i++) insert.run("x".repeat(4096));
      legacy.exec("INSERT INTO churn(body) VALUES ('survives-transition')");
      legacy.exec("DELETE FROM churn WHERE rowid <= 2000");
      legacy.exec("PRAGMA wal_checkpoint(TRUNCATE)");
    } finally {
      legacy.close();
    }
    const before = new DatabaseSync(dbPath, { readOnly: true });
    const legacyFreelist = (
      before.prepare("PRAGMA freelist_count").get() as { freelist_count: number }
    ).freelist_count;
    const legacyPageCount = (before.prepare("PRAGMA page_count").get() as { page_count: number })
      .page_count;
    const legacySize = (await fs.stat(dbPath)).size;
    before.close();
    expect(legacyFreelist).toBeGreaterThan(1500);

    await Effect.runPromise(
      Effect.gen(function* () {
        const sql = yield* SqlClient.SqlClient;
        const [autoVacuum] = yield* sql<{ readonly auto_vacuum: number }>`
          PRAGMA auto_vacuum;
        `;
        const [freelist] = yield* sql<{ readonly freelist_count: number }>`
          PRAGMA freelist_count;
        `;
        const [pageCount] = yield* sql<{ readonly page_count: number }>`
          PRAGMA page_count;
        `;
        expect(autoVacuum?.auto_vacuum).toBe(2);
        expect(freelist?.freelist_count).toBe(0);
        expect(pageCount?.page_count ?? 0).toBeLessThan(legacyPageCount);
        const rows = yield* sql<{ readonly body: string }>`
          SELECT body FROM churn
        `;
        expect(rows.map((row) => row.body)).toEqual(["survives-transition"]);
      }).pipe(
        Effect.provide(makeSqlitePersistenceLive(dbPath).pipe(Layer.provide(NodeServices.layer))),
      ),
    );

    const after = (await fs.stat(dbPath)).size;
    expect(after).toBeLessThan(legacySize);
  });

  it("creates new databases in incremental auto-vacuum so bounded passes reclaim the freelist", async () => {
    const dbPath = await makeDbPath();

    await Effect.runPromise(
      Effect.gen(function* () {
        const sql = yield* SqlClient.SqlClient;
        const [autoVacuum] = yield* sql<{ readonly auto_vacuum: number }>`
          PRAGMA auto_vacuum;
        `;
        expect(autoVacuum?.auto_vacuum).toBe(2);

        yield* sql`CREATE TABLE churn(body TEXT NOT NULL)`;
        for (let i = 0; i < 500; i++) {
          yield* sql`INSERT INTO churn(body) VALUES (${"x".repeat(4096)})`;
        }
        yield* sql`DELETE FROM churn`;
        const [grown] = yield* sql<{ readonly freelist_count: number }>`
          PRAGMA freelist_count;
        `;
        expect(grown?.freelist_count ?? 0).toBeGreaterThan(0);

        yield* sql`PRAGMA incremental_vacuum(100);`;
        const [drained] = yield* sql<{ readonly freelist_count: number }>`
          PRAGMA freelist_count;
        `;
        expect(drained?.freelist_count ?? 0).toBeLessThan(grown?.freelist_count ?? 0);
      }).pipe(
        Effect.provide(makeSqlitePersistenceLive(dbPath).pipe(Layer.provide(NodeServices.layer))),
      ),
    );
  });
});
