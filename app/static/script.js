// ================================================================
// script.js – Bank Fraud Detection Interface
// Auteur : Papa Malick NDIAYE | Master DSGL – UADB
// ================================================================

// ── Features du dataset ─────────────────────────────────────────
const FEATURES = [
  { key: "Gender",                   label: "Genre" },
  { key: "Age",                      label: "Âge" },
  { key: "HouseTypeID",              label: "Type de logement" },
  { key: "ContactAvaliabilityID",    label: "Disponibilité contact" },
  { key: "HomeCountry",              label: "Pays de résidence" },
  { key: "AccountNo",                label: "Numéro de compte" },
  { key: "CardExpiryDate",           label: "Expiration carte" },
  { key: "TransactionAmount",        label: "Montant transaction" },
  { key: "TransactionCountry",       label: "Pays transaction" },
  { key: "LargePurchase",            label: "Gros achat (0/1)" },
  { key: "ProductID",                label: "ID Produit" },
  { key: "CIF",                      label: "CIF Client" },
  { key: "TransactionCurrencyCode",  label: "Code devise" },
];

// ── Navbar scroll ────────────────────────────────────────────────
window.addEventListener("scroll", () => {
  document.getElementById("navbar")
    .classList.toggle("scrolled", window.scrollY > 30);
});

// ── Génération du formulaire ─────────────────────────────────────
function genererFormulaire() {
  const grid = document.getElementById("form-grid");
  FEATURES.forEach(({ key, label }) => {
    const div = document.createElement("div");
    div.className = "form-group";
    div.innerHTML = `
      <label for="f_${key}">${label}</label>
      <input type="number" id="f_${key}" step="any" placeholder="0" required />
    `;
    grid.appendChild(div);
  });
}

// ── Chargement des métriques ─────────────────────────────────────
async function chargerMetriques() {
  try {
    const res  = await fetch("/metrics");
    const data = await res.json();
    if (!data.meilleur_modele) return;

    const meilleur = data.meilleur_modele;
    const m        = data.resultats[meilleur];

    // Hero stat
    document.getElementById("hero-acc").textContent =
      (m.accuracy * 100).toFixed(1) + "%";

    // Overview cards
    document.getElementById("best-model-name").textContent = meilleur;
    document.getElementById("m-acc").textContent  = (m.accuracy  * 100).toFixed(1) + "%";
    document.getElementById("m-prec").textContent = (m.precision * 100).toFixed(1) + "%";
    document.getElementById("m-rec").textContent  = (m.recall    * 100).toFixed(1) + "%";
    document.getElementById("m-f1").textContent   = (m.f1_score  * 100).toFixed(1) + "%";

    // Tableau des modèles
    const tbody = document.getElementById("models-tbody");
    tbody.innerHTML = "";
    for (const [nom, met] of Object.entries(data.resultats)) {
      const estMeilleur = nom === meilleur;
      const tr = document.createElement("tr");
      if (estMeilleur) tr.className = "best-row";
      tr.innerHTML = `
        <td>
          ${nom}
          ${estMeilleur ? '<span class="badge-best">Meilleur</span>' : ""}
        </td>
        <td>${(met.accuracy  * 100).toFixed(2)}%</td>
        <td>${(met.precision * 100).toFixed(2)}%</td>
        <td>${(met.recall    * 100).toFixed(2)}%</td>
        <td>${(met.f1_score  * 100).toFixed(2)}%</td>
        <td>${estMeilleur ? "✅ Déployé" : "—"}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch {
    document.getElementById("best-model-name").textContent = "Lancez python main.py d'abord";
  }
}

// ── Prédiction ───────────────────────────────────────────────────
document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const btn      = document.getElementById("btn-predict");
  const idle     = document.getElementById("result-idle");
  const output   = document.getElementById("result-output");
  const badge    = document.getElementById("result-badge");
  const label    = document.getElementById("result-label");
  const probaFill = document.getElementById("proba-fill");
  const probaPct  = document.getElementById("proba-pct");

  // Lecture des valeurs
  const features = FEATURES.map(({ key }) => {
    const val = parseFloat(document.getElementById(`f_${key}`).value);
    return isNaN(val) ? 0 : val;
  });

  // UI loading
  btn.innerHTML  = '<i class="fas fa-circle-notch fa-spin"></i> Analyse…';
  btn.disabled   = true;
  idle.classList.add("hidden");
  output.classList.add("hidden");

  try {
    const res  = await fetch("/predict", {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ features })
    });
    const data = await res.json();

    if (data.erreur) {
      idle.classList.remove("hidden");
      alert("Erreur : " + data.erreur);
    } else {
      const estFraude = data.prediction === 1;
      const cls       = estFraude ? "fraude" : "normal";

      badge.textContent  = estFraude ? "🚨 Alerte" : "✅ Sûre";
      badge.className    = `result-badge ${cls}`;
      label.textContent  = estFraude ? "Fraude détectée" : "Transaction normale";
      label.className    = `result-label ${cls}`;

      probaPct.textContent    = data.probabilite + "%";
      probaFill.className     = `proba-bar-fill ${cls}`;
      probaFill.style.width   = "0%";

      output.classList.remove("hidden");

      // Animation barre
      requestAnimationFrame(() => {
        setTimeout(() => {
          probaFill.style.width = data.probabilite + "%";
        }, 50);
      });
    }
  } catch {
    idle.classList.remove("hidden");
    alert("Impossible de joindre l'API.");
  }

  btn.innerHTML = '<i class="fas fa-search"></i> Analyser la transaction';
  btn.disabled  = false;
});

// ── Reset ────────────────────────────────────────────────────────
document.getElementById("btn-reset").addEventListener("click", () => {
  document.getElementById("result-output").classList.add("hidden");
  document.getElementById("result-idle").classList.remove("hidden");
  document.getElementById("predict-form").reset();
});

// ── Scroll fluide ────────────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener("click", e => {
    const target = document.querySelector(a.getAttribute("href"));
    if (target) {
      e.preventDefault();
      window.scrollTo({
        top     : target.getBoundingClientRect().top + window.scrollY - 60,
        behavior: "smooth"
      });
    }
  });
});

// ── Init ─────────────────────────────────────────────────────────
genererFormulaire();
chargerMetriques();
