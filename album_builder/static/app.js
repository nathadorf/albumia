"use strict";

// ------------------------------------------------------------------ outils
const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem("albumia." + k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem("albumia." + k, JSON.stringify(v)); } catch { /* navigation privée */ } },
};

async function api(path, { method = "GET", body, form } = {}) {
  const opts = { method };
  if (form) { opts.body = form; }
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers = { "Content-Type": "application/json" }; }
  if (method === "GET" && !form && body === undefined) opts.method = "GET";
  let res;
  try { res = await fetch(path, opts); }
  catch {  // le serveur ne tourne plus (fenêtre fermée, PC redémarré…)
    const err = new Error("Albumia n'est pas lancé. Double-clique sur Albumia.bat (ou ouvre le dossier dans VS Code), puis recharge cette page.");
    err.code = "offline";
    throw err;
  }
  let data = null;
  try { data = await res.json(); } catch { /* réponse vide */ }
  if (!res.ok) {
    const err = new Error((data && data.error) || "Albumia ne répond plus. Relance-le depuis VS Code.");
    err.code = data && data.code;
    throw err;
  }
  return data;
}

async function waitJob(job, onTick) {
  while (job.state === "running") {
    onTick && onTick(job);
    await sleep(700);
    job = await api(`/api/jobs/${job.id}`);
  }
  onTick && onTick(job);
  return job;
}

function show(id) {
  $("#toast").hidden = true;
  document.querySelectorAll(".screen").forEach((s) => (s.hidden = s.id !== id));
  window.scrollTo({ top: 0 });
}

let toastTimer;
function toast(text, action) {
  const t = $("#toast");
  t.replaceChildren(el("span", {}, text));
  if (action) t.append(el("button", { onclick: () => { action.run(); t.hidden = true; } }, action.label));
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), action ? 6000 : 3500);
}

const MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"];
const fmtDate = (iso) => { const d = new Date(iso); return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`; };
const nf = new Intl.NumberFormat("fr-FR");

// ------------------------------------------------------------------ état
const state = {
  status: null,
  people: [],
  selected: new Set(store.get("selected", [])),
  intent: null,
  dirty: new Set(),          // champs modifiés à la main : la description ne les écrase plus
  roles: {},                 // prénom -> "must" | "bonus"
  folder: { link: "", ok: false, name: "" },
  plan: null, slug: null, photos: [], reserve: [], removed: [],
  buildJob: null,
};

// ------------------------------------------------------------------ démarrage
async function boot() {
  try {
    state.status = await api("/api/status");
  } catch (e) {
    document.body.replaceChildren(el("main", {}, el("div", { class: "card hero" }, el("h1", {}, "Albumia n'est pas lancé"), el("p", { class: "lead" }, "Ouvre le dossier Albumia dans VS Code : il se lance tout seul."))));
    return;
  }
  const s = state.status;
  if (!s.connected) {
    $("#setup-credentials").hidden = s.credentials;
    $("#setup-login").hidden = !s.credentials;
    show("screen-setup");
    return;
  }
  $("#account").hidden = false;
  $("#account").textContent = s.email ? `Connecté : ${s.email}` : "Google Drive connecté";
  $("#claude-toggle").hidden = !s.claude;
  await loadPeople();
  loadPastAlbums();
  restoreForm();
  show("screen-form");
}

// ------------------------------------------------------------------ installation / connexion
function setupDrop(label, onFiles) {
  const input = label.querySelector("input");
  input.addEventListener("change", () => { if (input.files.length) onFiles([...input.files]); input.value = ""; });
  label.addEventListener("dragover", (e) => { e.preventDefault(); label.classList.add("over"); });
  label.addEventListener("dragleave", () => label.classList.remove("over"));
  label.addEventListener("drop", (e) => {
    e.preventDefault(); label.classList.remove("over");
    if (e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]);
  });
}

setupDrop($("#cred-drop"), async ([file]) => {
  const form = new FormData(); form.append("file", file);
  try { await api("/api/credentials", { method: "POST", form }); toast("Fichier enregistré. Connecte-toi maintenant."); boot(); }
  catch (e) { toast(e.message); }
});

async function doLogin(full = false) {
  $("#login-hint").hidden = false;
  const btn = $("#btn-login"); btn.disabled = true; btn.textContent = "En attente de Google…";
  try {
    const job = await waitJob(await api("/api/login", { method: "POST", body: { full } }));
    if (job.state !== "done") throw new Error(job.error ? job.error.message : "Connexion interrompue.");
    return true;
  } catch (e) {
    toast(e.message);
    return false;
  } finally {
    btn.disabled = false; btn.textContent = "Se connecter à Google";
  }
}
$("#btn-login").addEventListener("click", async () => { if (await doLogin()) boot(); });

// ------------------------------------------------------------------ personnes
async function loadPeople() {
  state.people = await api("/api/people");
  const names = new Set(state.people.map((p) => p.name));
  state.selected = new Set([...state.selected].filter((n) => names.has(n)));
  renderPeople();
}

function renderPeople() {
  const box = $("#people");
  box.replaceChildren(
    ...state.people.map((p) => el("div", {
      class: "person" + (state.selected.has(p.name) ? " on" : ""), title: "Choisir / retirer",
      onclick: () => togglePerson(p.name),
    },
      el("div", { class: "avatar" }, p.photos[0] ? el("img", { src: p.photos[0], alt: "" }) : "🙂"),
      el("div", { class: "pname" }, p.name),
      el("button", { type: "button", class: "pedit", onclick: (e) => { e.stopPropagation(); openPerson(p.name); } }, `${p.photos.length} photo${p.photos.length > 1 ? "s" : ""}`),
    )),
    el("div", { class: "person add", onclick: openFaces },
      el("div", { class: "avatar" }, "+"),
      el("div", { class: "pname" }, "Ajouter")),
    el("div", { class: "person add", onclick: openGroups, title: "Albumia te montre les visages qu'il a vus dans tes photos" },
      el("div", { class: "avatar" }, "🔎"),
      el("div", { class: "pname" }, "Retrouver")),
  );
  const datalist = $("#known-names");
  datalist.replaceChildren(...state.people.map((p) => el("option", { value: p.name })));
}

function togglePerson(name) {
  if (state.selected.has(name)) state.selected.delete(name); else state.selected.add(name);
  store.set("selected", [...state.selected]);
  state.dirty.delete("roles");
  renderPeople();
  interpretSoon(0);
}

// --- fenêtre « ajouter des personnes »
let pendingFaces = [];
function openFaces() {
  pendingFaces = [];
  $("#faces-list").replaceChildren();
  $("#faces-tip").hidden = true;
  $("#btn-faces-save").disabled = true;
  $("#dlg-faces").showModal();
}

setupDrop($("#faces-drop"), async (files) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  $("#faces-busy").hidden = false;
  try {
    const results = await api("/api/faces", { method: "POST", form });
    for (const r of results) {
      if (r.error) { $("#faces-list").append(el("div", { class: "face-error" }, `${r.file} : ${r.error}`)); continue; }
      if (!r.faces.length) { $("#faces-list").append(el("div", { class: "face-error" }, `${r.file} : je ne vois aucun visage. Essaie une photo plus nette, de face.`)); continue; }
      for (const f of r.faces) addPendingFace(f);
    }
  } catch (e) {
    $("#faces-list").append(el("div", { class: "face-error" }, e.message));
  } finally {
    $("#faces-busy").hidden = true;
    $("#faces-tip").hidden = !pendingFaces.length;
    refreshFacesSave();
  }
});

function addPendingFace(f) {
  const item = { token: f.token, name: f.guess || "", skip: false };
  pendingFaces.push(item);
  const input = el("input", { type: "text", list: "known-names", placeholder: "Qui est-ce ?", value: item.name,
    oninput: (e) => { item.name = e.target.value; refreshFacesSave(); },
    onkeydown: (e) => { if (e.key === "Enter") e.preventDefault(); } });  // Entrée ne doit pas fermer la fenêtre
  const card = el("div", { class: "face" },
    el("img", { src: f.preview, alt: "" }),
    input,
    el("button", { type: "button", class: "x", title: "Ignorer ce visage", onclick: () => {
      item.skip = !item.skip; card.classList.toggle("skip", item.skip); input.disabled = item.skip; refreshFacesSave();
    } }, "✕"));
  $("#faces-list").append(card);
}

function refreshFacesSave() {
  $("#btn-faces-save").disabled = !pendingFaces.some((f) => !f.skip && f.name.trim());
}

$("#dlg-faces").addEventListener("close", async () => {
  if ($("#dlg-faces").returnValue !== "save") return;
  const items = pendingFaces.filter((f) => !f.skip && f.name.trim()).map((f) => ({ token: f.token, name: f.name.trim() }));
  try {
    const res = await api("/api/faces/save", { method: "POST", body: { items } });
    state.people = res.people;
    items.forEach((i) => {
      const p = state.people.find((x) => x.name.toLowerCase() === i.name.toLowerCase());
      if (p) state.selected.add(p.name);
    });
    store.set("selected", [...state.selected]);
    await loadPeople();
    toast(`${res.saved} visage${res.saved > 1 ? "s" : ""} enregistré${res.saved > 1 ? "s" : ""}.`);
    interpretSoon(0);
  } catch (e) { toast(e.message); }
});

// --- fenêtre « une personne »
let openedPerson = null;
function openPerson(name) {
  openedPerson = name;
  const p = state.people.find((x) => x.name === name);
  $("#pp-title").textContent = name;
  $("#pp-birth").value = p.birth || "";
  $("#pp-photos").replaceChildren(...p.photos.map((src) => el("div", { class: "face" },
    el("img", { src, alt: "" }),
    el("button", { type: "button", class: "x", title: "Retirer ce visage", onclick: async () => {
      state.people = await api(`/api/people/${encodeURIComponent(name)}/${encodeURIComponent(src.split("/").pop())}`, { method: "DELETE" });
      renderPeople();
      if (!state.people.find((x) => x.name === name)) $("#dlg-person").close(); else openPerson(name);
    } }, "✕"))));
  if (!$("#dlg-person").open) $("#dlg-person").showModal();
}
$("#pp-birth").addEventListener("change", async () => {
  try {
    state.people = await api(`/api/people/${encodeURIComponent(openedPerson)}`, { method: "PUT", body: { birth: $("#pp-birth").value || null } });
    toast("Date de naissance enregistrée.");
    interpretSoon(0);
  } catch (e) { toast(e.message); }
});

// --- « Qui est sur tes photos ? » : groupes de visages trouvés dans les photos déjà analysées
async function openGroups() {
  const list = $("#groups-list");
  list.replaceChildren(el("p", { class: "hint" }, "Je regroupe les visages de tes photos…"));
  $("#dlg-groups").showModal();
  try {
    const groups = await api("/api/face-groups");
    if (!groups.length) { list.replaceChildren(el("p", { class: "hint" }, "Je ne trouve pas de nouveau visage : tout le monde est déjà connu.")); return; }
    list.replaceChildren(...groups.map((g) => {
      const input = el("input", { type: "text", list: "known-names", placeholder: "Qui est-ce ?" });
      const save = el("button", { type: "button", class: "btn primary small", onclick: async () => {
        if (!input.value.trim()) { input.focus(); return; }
        save.disabled = true;
        try {
          const res = await api(`/api/face-groups/${g.id}`, { method: "POST", body: { name: input.value.trim() } });
          state.people = res.people; renderPeople();
          card.replaceChildren(el("p", { class: "ok-line" }, `✓ ${res.saved} visages ajoutés à ${input.value.trim()}`));
        } catch (e) { toast(e.message); save.disabled = false; }
      } }, "Enregistrer");
      const card = el("div", { class: "group" },
        el("div", { class: "group-faces" }, ...g.previews.map((src) => el("img", { src, alt: "" }))),
        el("div", { class: "sub" }, `${g.count} photos`),
        el("div", { class: "group-row" }, input, save));
      return card;
    }));
  } catch (e) {
    list.replaceChildren(el("p", { class: "face-error" }, e.message));
  }
}

$("#dlg-person").addEventListener("close", async () => {
  if ($("#dlg-person").returnValue !== "delete" || !openedPerson) return;
  if (!confirm(`Oublier ${openedPerson} ? Ses photos de référence seront effacées de cet ordinateur (rien n'est touché dans Drive).`)) return;
  state.people = await api(`/api/people/${encodeURIComponent(openedPerson)}`, { method: "DELETE" });
  state.selected.delete(openedPerson);
  store.set("selected", [...state.selected]);
  renderPeople();
  interpretSoon(0);
});

// ------------------------------------------------------------------ description
const EXAMPLES = [
  "50 photos de l'année 2026",
  "Une trentaine de photos de l'été 2025, plein de sourires",
  "Les plus beaux portraits de décembre",
  "40 photos de nous tous ensemble en 2025",
];
$("#examples").replaceChildren(...EXAMPLES.map((x) => el("button", { type: "button", class: "chip", onclick: () => {
  $("#prompt").value = x; state.dirty.clear(); store.set("prompt", x); interpretSoon(0); $("#prompt").focus();
} }, x)));

let interpretTimer;
function interpretSoon(delay = 350) {
  clearTimeout(interpretTimer);
  interpretTimer = setTimeout(runInterpret, delay);
}

async function runInterpret() {
  const prompt = $("#prompt").value;
  let it;
  try { it = await api("/api/interpret", { method: "POST", body: { prompt, selected: [...state.selected] } }); }
  catch { return; }
  state.intent = it;
  // les personnes citées dans le texte sont cochées automatiquement
  let changed = false;
  for (const n of [...it.must, ...it.bonus]) if (!state.selected.has(n)) { state.selected.add(n); changed = true; }
  if (changed) { store.set("selected", [...state.selected]); renderPeople(); }

  const set = (key, input, value, prop = "value") => { if (!state.dirty.has(key)) input[prop] = value; };
  set("name", $("#f-name"), it.name);
  set("count", $("#f-count"), it.count);
  set("from", $("#f-from"), it.date_from || "");
  set("to", $("#f-to"), it.date_to || "");
  set("smiles", $("#f-smiles"), it.favor_smiles, "checked");
  set("portraits", $("#f-portraits"), it.faces_only, "checked");
  if (!state.dirty.has("roles")) {
    state.roles = {};
    it.must.forEach((n) => (state.roles[n] = "must"));
    it.bonus.forEach((n) => (state.roles[n] = "bonus"));
  }
  renderRoles();
}

function renderRoles() {
  const box = $("#roles");
  const names = [...state.selected];
  if (!names.length) {
    box.replaceChildren(el("span", { class: "sub" }, "Personne en particulier : je choisis les plus belles photos, tout simplement."));
    return;
  }
  box.replaceChildren(...names.map((n) => {
    const role = state.roles[n] || "must";
    const p = state.people.find((x) => x.name === n);
    const pick = (r) => () => { state.roles[n] = r; state.dirty.add("roles"); renderRoles(); };
    return el("div", { class: "role" },
      p && p.photos[0] ? el("img", { src: p.photos[0], alt: "" }) : null,
      el("b", {}, n),
      el("span", { class: "seg" },
        el("button", { type: "button", class: role === "must" ? "on" : "", onclick: pick("must") }, "sur chaque photo"),
        el("button", { type: "button", class: role === "bonus" ? "on" : "", onclick: pick("bonus") }, "si possible")));
  }));
}

$("#prompt").addEventListener("input", () => {
  // une nouvelle description reprend la main, sauf sur le nom choisi à la main
  ["count", "from", "to", "smiles", "portraits", "roles"].forEach((k) => state.dirty.delete(k));
  store.set("prompt", $("#prompt").value);
  interpretSoon();
});
for (const [key, id] of [["name", "#f-name"], ["count", "#f-count"], ["from", "#f-from"], ["to", "#f-to"], ["smiles", "#f-smiles"], ["portraits", "#f-portraits"]]) {
  $(id).addEventListener("input", () => state.dirty.add(key));
}

// ------------------------------------------------------------------ dossier Drive
let folderTimer, folderSeq = 0;
$("#folder").addEventListener("input", () => {
  clearTimeout(folderTimer);
  state.folder = { link: $("#folder").value.trim(), ok: false, name: "" };
  updateBuildButton();
  folderTimer = setTimeout(checkFolder, 400);
});

async function checkFolder() {
  const link = $("#folder").value.trim();
  const status = $("#folder-status");
  status.className = "folder-status";
  if (!link) { status.textContent = ""; return; }
  status.textContent = "Je vérifie le dossier…";
  const seq = ++folderSeq;
  try {
    const f = await api("/api/folder", { method: "POST", body: { link } });
    if (seq !== folderSeq) return;
    state.folder = { link, ok: true, name: f.name };
    status.className = "folder-status ok";
    status.textContent = `✓ Dossier trouvé : « ${f.name} »`;
    store.set("folder", link);
  } catch (e) {
    if (seq !== folderSeq) return;
    status.className = "folder-status bad";
    status.textContent = e.message;
  }
  updateBuildButton();
}

function updateBuildButton() {
  $("#btn-build").disabled = !state.folder.ok;
  $("#build-hint").textContent = state.folder.ok
    ? "Rien n'est créé dans ton Drive avant que tu aies validé la sélection."
    : "Colle d'abord le lien du dossier de photos (étape 3).";
}

function restoreForm() {
  if (!$("#prompt").value) $("#prompt").value = store.get("prompt", "");
  if (!$("#folder").value) $("#folder").value = store.get("folder", "");
  renderRoles();
  runInterpret();
  if ($("#folder").value) checkFolder(); else updateBuildButton();
}

// ------------------------------------------------------------------ albums précédents
async function loadPastAlbums() {
  let albums = [];
  try { albums = await api("/api/albums"); } catch { /* pas grave */ }
  $("#past-albums").hidden = !albums.length;
  $("#past-row").replaceChildren(...albums.map((a) => {
    const inner = [
      a.cover ? el("img", { src: a.cover, alt: "", loading: "lazy" }) : el("img", { alt: "" }),
      el("div", {}, el("b", {}, a.name), a.drive_url ? `${a.count} photos · dans Drive` : "à valider"),
    ];
    return a.drive_url
      ? el("a", { class: "past-item", href: a.drive_url, target: "_blank", rel: "noopener" }, ...inner)
      : el("a", { class: "past-item", href: "#", onclick: (e) => { e.preventDefault(); openReview(a.slug); } }, ...inner);
  }));
}

// ------------------------------------------------------------------ préparation
const STEP_LABELS = ["Recherche des photos dans Drive", "Reconnaissance des visages", "Analyse de chaque photo", "Choix des meilleures", "Préparation de l'aperçu"];

function buildRequest(extra = {}) {
  const names = [...state.selected];
  return {
    prompt: $("#prompt").value,
    folder: state.folder.link,
    name: $("#f-name").value,
    count: Number($("#f-count").value) || 50,
    date_from: $("#f-from").value || null,
    date_to: $("#f-to").value || null,
    must: names.filter((n) => (state.roles[n] || "must") === "must"),
    bonus: names.filter((n) => state.roles[n] === "bonus"),
    favor_smiles: $("#f-smiles").checked,
    faces_only: $("#f-portraits").checked,
    use_claude: $("#f-claude").checked,
    ...extra,
  };
}

$("#btn-build").addEventListener("click", () => startBuild());

async function startBuild(extra = {}) {
  show("screen-progress");
  $("#p-title").textContent = "Je prépare ton album…";
  $("#btn-cancel").hidden = false;
  const samples = [];
  try {
    let job = await api("/api/build", { method: "POST", body: buildRequest(extra) });
    state.buildJob = job;
    job = await waitJob(job, (j) => renderProgress(j, samples));
    if (job.state === "cancelled") { show("screen-form"); toast("Préparation annulée. Le travail déjà fait est gardé pour la prochaine fois."); return; }
    if (job.state === "error") return buildError(job.error);
    await openReview(job.result.slug);
  } catch (e) {
    buildError({ message: e.message, code: e.code });
  }
}

function renderProgress(job, samples) {
  $("#p-label").textContent = job.label;
  const per = 100 / job.steps;
  let pct = Math.max(0, job.step - 1) * per;
  let detail = job.notes.length ? job.notes[job.notes.length - 1] : "";
  if (job.total) {
    pct += per * (job.done / job.total);
    const now = Date.now();
    samples.push([now, job.done]);
    while (samples.length > 1 && now - samples[0][0] > 30000) samples.shift();
    detail = `${nf.format(job.done)} / ${nf.format(job.total)} photos`;
    if (samples.length > 3) {
      const [t0, d0] = samples[0];
      const rate = (job.done - d0) / ((now - t0) / 1000);
      if (rate > 0) {
        const left = (job.total - job.done) / rate;
        detail += left > 90 ? ` · encore environ ${Math.round(left / 60)} min` : left > 10 ? " · encore moins de 2 min" : " · presque fini";
      }
    }
  }
  $("#p-bar").style.width = `${Math.min(100, pct)}%`;
  $("#p-detail").textContent = detail;
  $("#p-steps").replaceChildren(...STEP_LABELS.map((label, i) => el("li", {
    class: i + 1 < job.step ? "done" : i + 1 === job.step ? "now" : "todo",
  }, label)));
}

$("#btn-cancel").addEventListener("click", async () => {
  if (!state.buildJob) return;
  $("#btn-cancel").hidden = true;
  $("#p-title").textContent = "J'arrête…";
  try { await api(`/api/jobs/${state.buildJob.id}/cancel`, { method: "POST" }); } catch { /* déjà fini */ }
});

function buildError(error) {
  show("screen-form");
  const code = error && error.code;
  const msg = (error && error.message) || "Une erreur est survenue.";
  if (code === "no_match") {
    if (confirm(`${msg}\n\nVeux-tu que je réessaie en étant moins strict sur la reconnaissance des visages ?`)) {
      startBuild({ lenient: true });
    }
    return;
  }
  if (code === "needs_login") { boot(); return; }
  alert(msg);
}

// ------------------------------------------------------------------ relecture
async function openReview(slug) {
  const plan = await api(`/api/plans/${encodeURIComponent(slug)}`);
  state.plan = plan; state.slug = slug;
  state.photos = plan.photos.slice();
  state.reserve = (plan.reserve || []).slice();
  state.removed = [];
  $("#r-name").value = plan.album_name;
  const s = plan.stats || {};
  const wanted = plan.settings && plan.settings.count;
  let stats = `${plan.photos.length} photos choisies parmi ${nf.format(s.analyzed ?? s.listed ?? 0)} regardées.`;
  if (wanted && plan.photos.length < wanted) stats += ` Je n'en ai trouvé que ${plan.photos.length} qui correspondent à ta demande.`;
  if (s.no_thumbnail) stats += ` ${s.no_thumbnail} photo(s) sans aperçu Drive ont été ignorées.`;
  $("#r-stats").textContent = stats;
  renderReview();
  show("screen-review");
}

const thumb = (p) => `/plans/${encodeURIComponent(state.slug)}/thumbs/${p.id}.jpg`;
const monthKey = (iso) => iso.slice(0, 7);

function renderReview(freshId) {
  const groups = new Map();
  for (const p of [...state.photos].sort((a, b) => a.date.localeCompare(b.date))) {
    const k = monthKey(p.date);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(p);
  }
  const grid = $("#r-grid");
  grid.replaceChildren();
  for (const [k, items] of groups) {
    const [y, m] = k.split("-").map(Number);
    grid.append(el("div", { class: "month" }, `${MONTHS[m - 1]} ${y}`));
    grid.append(el("div", { class: "grid" }, ...items.map((p) => el("div", {
      class: "tile" + (p.id === freshId ? " fresh" : ""), onclick: () => openLightbox(p),
    },
      el("img", { src: thumb(p), alt: p.name, loading: "lazy" }),
      p.starred ? el("span", { class: "star", title: "Étoilée dans Drive" }, "★") : null,
      el("button", { type: "button", class: "x", title: "Retirer cette photo", onclick: (e) => { e.stopPropagation(); askReason(p, e.currentTarget); } }, "✕"),
      el("button", { type: "button", class: "swap", title: "Voir d'autres photos de ce moment", onclick: (e) => { e.stopPropagation(); showAlternatives(p, e.currentTarget); } }, "⇄"),
    ))));
  }
  const n = state.photos.length;
  $("#r-count").textContent = `${n} photo${n > 1 ? "s" : ""}`;
  $("#btn-apply").disabled = n === 0;
}

// --- petite fenêtre accrochée à une photo (raison du retrait, autres photos du moment)
function popover(anchor, ...children) {
  closePopover();
  const box = el("div", { class: "popover", onclick: (e) => e.stopPropagation() }, ...children);
  const r = anchor.getBoundingClientRect();
  box.style.top = `${window.scrollY + r.bottom + 6}px`;
  box.style.left = `${Math.max(8, Math.min(window.scrollX + r.right - 260, document.documentElement.clientWidth - 268))}px`;
  document.body.append(box);
  setTimeout(() => document.addEventListener("click", closePopover, { once: true }));
}
function closePopover() { document.querySelectorAll(".popover").forEach((x) => x.remove()); }

const REASONS = [
  ["pas-belle", "Pas belle (floue, sombre, ratée)"],
  ["doublon", "Déjà une photo de ce moment"],
  ["personnes", "Pas les bonnes personnes"],
  ["gout", "Je ne l'aime pas"],
];
function askReason(p, anchor) {
  popover(anchor, el("div", { class: "pop-title" }, "Pourquoi la retirer ? Albumia s'en souviendra."),
    ...REASONS.map(([code, label]) => el("button", { type: "button", class: "pop-item", onclick: () => { closePopover(); removePhoto(p, code); } }, label)));
}

// autres photos du même moment (même jour, sinon même mois), les mieux notées d'abord
function alternativesFor(p) {
  const day = p.date.slice(0, 10);
  const same = state.reserve.filter((x) => x.date.slice(0, 10) === day);
  const month = state.reserve.filter((x) => monthKey(x.date) === monthKey(p.date) && !same.includes(x));
  return [...same, ...month].slice(0, 3);
}
function showAlternatives(p, anchor) {
  const alts = alternativesFor(p);
  if (!alts.length) { toast("Pas d'autre bonne photo de ce moment."); return; }
  popover(anchor, el("div", { class: "pop-title" }, "Remplacer par :"),
    el("div", { class: "pop-alts" }, ...alts.map((a) => el("img", { src: thumb(a), alt: a.name, title: fmtDate(a.date),
      onclick: () => { closePopover(); replacePhoto(p, a, "gout"); } }))));
}

function replacePhoto(p, replacement, reason) {
  const idx = state.photos.indexOf(p);
  if (idx < 0) return;
  if (replacement) state.reserve.splice(state.reserve.indexOf(replacement), 1);
  state.photos.splice(idx, 1, ...(replacement ? [replacement] : []));
  state.removed.push({ id: p.id, reason });
  renderReview(replacement && replacement.id);
  toast(replacement ? "Photo remplacée." : "Photo retirée.", {
    label: "Annuler",
    run: () => {
      state.removed = state.removed.filter((r) => r.id !== p.id);
      const j = replacement ? state.photos.indexOf(replacement) : -1;
      if (j >= 0) { state.photos.splice(j, 1, p); state.reserve.unshift(replacement); }
      else state.photos.push(p);
      renderReview();
    },
  });
}

function removePhoto(p, reason) {
  // remplaçante : la meilleure du même moment, sinon du même mois, sinon la meilleure tout court
  const replacement = alternativesFor(p)[0] || state.reserve[0] || null;
  replacePhoto(p, replacement, reason);
}

function openLightbox(p) {
  const lb = $("#lightbox");
  lb.querySelector("img").src = thumb(p);
  const people = p.people && p.people.length ? ` · ${p.people.join(", ")}` : "";
  lb.querySelector(".lb-caption").textContent = `${fmtDate(p.date)} · ${p.name}${people}${p.note ? " · " + p.note : ""}`;
  lb.hidden = false;
}
$("#lightbox").addEventListener("click", () => ($("#lightbox").hidden = true));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").hidden = true; });

$("#btn-back").addEventListener("click", () => { show("screen-form"); loadPastAlbums(); });

// ------------------------------------------------------------------ création dans Drive
$("#btn-apply").addEventListener("click", () => applyAlbum(false));

async function applyAlbum(full) {
  const btn = $("#btn-apply");
  btn.disabled = true; btn.textContent = "Création en cours…";
  try {
    const body = { slug: state.slug, ids: state.photos.map((p) => p.id), name: $("#r-name").value, full, removed: state.removed };
    const job = await waitJob(await api("/api/apply", { method: "POST", body }), (j) => {
      if (j.total) btn.textContent = `Création… ${j.done} / ${j.total}`;
    });
    if (job.state === "error") {
      if (job.error.code === "needs_full_scope" && confirm(`${job.error.message}\n\nUne page Google va s'ouvrir pour l'accepter. On y va ?`)) {
        if (await doLogin(true)) return applyAlbum(true);
        return;
      }
      throw new Error(job.error.message);
    }
    $("#d-text").textContent = `« ${job.result.name} » contient ${job.result.count} photos.`;
    $("#d-link").href = job.result.url;
    show("screen-done");
  } catch (e) {
    alert(e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Créer l'album dans mon Drive";
  }
}

$("#btn-again").addEventListener("click", () => {
  $("#prompt").value = ""; store.set("prompt", ""); state.dirty.clear();
  runInterpret(); loadPastAlbums(); show("screen-form");
});

boot();
