// Session persistence in IndexedDB.
// A session = the embedded trace of one recording:
//   { id, name, createdAt, modelId, points: [{x,y,t}], latents: [[z…]] }
// Latents are stored so any region can later be decoded back into motion.

const DB = 'motem-db', STORE = 'sessions';

function openDb() {
  return new Promise((res, rej) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id' });
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
}

async function tx(mode, fn) {
  const db = await openDb();
  return new Promise((res, rej) => {
    const t = db.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => { db.close(); res(out.result ?? out); };
    t.onerror = () => { db.close(); rej(t.error); };
  });
}

export const saveSession = (sess) => tx('readwrite', s => s.put(sess));
export const deleteSession = (id) => tx('readwrite', s => s.delete(id));

export async function listSessions() {
  const db = await openDb();
  return new Promise((res, rej) => {
    const req = db.transaction(STORE).objectStore(STORE).getAll();
    req.onsuccess = () => { db.close(); res(req.result.sort((a, b) => b.createdAt - a.createdAt)); };
    req.onerror = () => { db.close(); rej(req.error); };
  });
}

export async function getSession(id) {
  const db = await openDb();
  return new Promise((res, rej) => {
    const req = db.transaction(STORE).objectStore(STORE).get(id);
    req.onsuccess = () => { db.close(); res(req.result); };
    req.onerror = () => { db.close(); rej(req.error); };
  });
}
