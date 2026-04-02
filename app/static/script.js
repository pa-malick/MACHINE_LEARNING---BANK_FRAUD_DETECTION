// ================================================================
// script.js  –  Logique front-end de l'interface de prédiction
// Auteur : Papa Malick NDIAYE
// ================================================================

// Noms des 13 features dans l'ordre exact du dataset
const FEATURES = [
  "CustomerID", "Age", "Income", "AccountBalance",
  "NumTransactions", "NumLatePayments", "CreditScore",
  "LoanAmount", "LoanDuration", "NumCreditCards",
  "HasLoan", "HasMortgage", "TransactionFrequency"
];

// Nom lisible pour chaque feature
const LABELS = {
  CustomerID           : "ID Client",
  Age                  : "Âge",
  Income               : "Revenu",
  AccountBalance       : "Solde compte",
  NumTransactions      : "Nb transactions",
  NumLatePayments      : "Nb retards paiement",
  CreditScore          : "Score crédit",
  LoanAmount           : "Montant prêt",
  LoanDuration         : "Durée prêt (mois)",
  NumCreditCards       : "Nb cartes crédit",
  HasLoan              : "A un prêt (0/1)",
  HasMortgage          : "Hypothèque (0/1)",
  TransactionFrequency : "Fréquence transactions"
};

// ── Génération du formulaire ─────────────────────────────────────
function genererFormulaire() {
  const grid = document.getElementById("form-grid");
  FEATURES.forEach(nom => {
    const div = document.createElement("div");
    div.className = "form-group";
    div.innerHTML = `
      <label for="f_${nom}">${LABELS[nom] || nom}</label>
      <input type="number" id="f_${nom}" step="any" placeholder="0" required />
    `;
    grid.appendChild(div);
  });
}

// ── Chargement des métriques depuis l'API ────────────────────────
async function chargerMetriques() {
  try {
    const res  = await fetch("/metrics");
    const data = await res.json();

    if (!data.meilleur_modele) return;

    const meilleur = data.meilleur_modele;
    const m        = data.resultats[meilleur];

    document.getElementById("best-model-name").textContent = meilleur;
    document.getElementById("m-acc").textContent  = (m.accuracy  * 100).toFixed(1) + "%";
    document.getElementById("m-prec").textContent = (m.precision * 100).toFixed(1) + "%";
    document.getElementById("m-rec").textContent  = (m.recall    * 100).toFixed(1) + "%";
    document.getElementById("m-f1").textContent   = (m.f1_score  * 100).toFixed(1) + "%";

    // Tableau comparatif
    const tbody = document.getElementById("table-body");
    tbody.innerHTML = "";
    for (const [nom, met] of Object.entries(data.resultats)) {
      const tr = document.createElement("tr");
      if (nom === meilleur) tr.className = "best-row";
      tr.innerHTML = `
        <td>${nom}${nom === meilleur ? " 🏆" : ""}</td>
        <td>${(met.accuracy  * 100).toFixed(2)}%</td>
        <td>${(met.precision * 100).toFixed(2)}%</td>
        <td>${(met.recall    * 100).toFixed(2)}%</td>
        <td>${(met.f1_score  * 100).toFixed(2)}%</td>
      `;
      tbody.appendChild(tr);
    }
  } catch {
    document.getElementById("best-model-name").textContent =
      "Lancez d'abord : python main.py";
  }
}

// ── Soumission et prédiction ─────────────────────────────────────
document.getElementById("pred-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const btn    = document.getElementById("btn-submit");
  const result = document.getElementById("result-box");

  // Lecture des valeurs du formulaire
  const features = FEATURES.map(nom => {
    const val = parseFloat(document.getElementById(`f_${nom}`).value);
    return isNaN(val) ? 0 : val;
  });

  btn.textContent = "Analyse en cours…";
  btn.disabled    = true;
  result.className = "result hidden";

  try {
    const res  = await fetch("/predict", {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ features })
    });
    const data = await res.json();

    if (data.erreur) {
      result.textContent = "Erreur : " + data.erreur;
      result.className   = "result fraude";
    } else {
      result.innerHTML = `
        ${data.label}<br>
        <small style="font-weight:400;font-size:0.78rem;opacity:0.8;">
          Probabilité de fraude : ${data.probabilite} %
        </small>
      `;
      result.className = data.prediction === 1 ? "result fraude" : "result normal";
    }
  } catch {
    result.textContent = "Impossible de joindre l'API.";
    result.className   = "result fraude";
  }

  btn.textContent = "Analyser ↗";
  btn.disabled    = false;
});

// ── Init ─────────────────────────────────────────────────────────
genererFormulaire();
chargerMetriques();
