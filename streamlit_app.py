# ───────────────────────────────────────────────────────────
#  tactic_evaluator_app.py
#  ----------------------------------------------------------
#  Streamlit app to evaluate the output from your
#  “Marketing‑Tactic Text Classifier”:
#  • Aggregates word‑level hit‑rates at an ID level
#  • Computes precision, recall, and F1 for each tactic
#  ----------------------------------------------------------
#  REQUIREMENTS: pandas, streamlit 1.28+, (no external ML libs)
# ───────────────────────────────────────────────────────────
import ast
import math
import streamlit as st
import pandas as pd

st.set_page_config(page_title="📏 Tactic Classifier Evaluator", layout="wide")
st.title("📏 Tactic Classifier Evaluator")

# ───────────────────────────────────────────────────────────
#  Built‑in tactic dictionaries (same as first app)
# ───────────────────────────────────────────────────────────
DEFAULT_TACTICS = {
    "urgency_marketing":  ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof":       ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
    "Classic_Timeless_Luxury_style": [
        'elegance', 'heritage', 'sophistication', 'refined', 'timeless', 'grace',
        'legacy', 'opulence', 'bespoke', 'tailored', 'understated', 'prestige',
        'quality', 'craftsmanship', 'heirloom', 'classic', 'tradition', 'iconic',
        'enduring', 'rich', 'authentic', 'luxury', 'fine', 'pure', 'exclusive',
        'elite', 'mastery', 'immaculate', 'flawless', 'distinction', 'noble',
        'chic', 'serene', 'clean', 'minimal', 'poised', 'balanced', 'eternal',
        'neutral', 'subtle', 'grand', 'timelessness', 'tasteful', 'quiet', 'sublime'
    ]
}

# ───────────────────────────────────────────────────────────
#  1 ▸ UPLOAD PREDICTIONS CSV
# ───────────────────────────────────────────────────────────
pred_file = st.file_uploader(
    "📁 Upload *classified_results.csv* (predictions output from first app)",
    type="csv",
    key="pred_csv"
)

if pred_file:
    df_pred = pd.read_csv(pred_file)
    st.success(f"Loaded {len(df_pred):,} rows from predictions file.")
    st.dataframe(df_pred.head())

    # ───────────────────────────────────────────────────────
    #  2 ▸ SELECT COLUMN NAMES
    # ───────────────────────────────────────────────────────
    st.header("🔧 Column Mapping")

    id_col = st.selectbox(
        "ID column (optional – for aggregation)",
        options=["‑‑ none ‑‑"] + list(df_pred.columns),
        index=0
    )
    id_col = None if id_col == "‑‑ none ‑‑" else id_col

    text_col = st.selectbox(
        "Column containing cleaned text",
        options=list(df_pred.columns),
        index=list(df_pred.columns).index("cleaned") if "cleaned" in df_pred.columns else 0
    )

    pred_col = st.selectbox(
        "Column containing *predicted* categories list",
        options=list(df_pred.columns),
        index=list(df_pred.columns).index("categories") if "categories" in df_pred.columns else 0
    )

    # Ground‑truth labels may live in same file or separate upload
    with st.expander("Add ground‑truth labels (optional)"):
        has_gt = st.checkbox("Ground‑truth labels are in the predictions file", value=False)
        if has_gt:
            gt_col = st.selectbox(
                "Column containing *true* categories list",
                options=list(df_pred.columns)
            )
            df_gt = df_pred[[gt_col]]
        else:
            gt_file = st.file_uploader(
                "📁 Upload a separate CSV with ground‑truth labels (must share ID)",
                type="csv",
                key="gt_csv"
            )
            if gt_file and id_col:
                df_gt = pd.read_csv(gt_file)
                st.success(f"Loaded {len(df_gt):,} rows of ground‑truth labels.")
                gt_col = st.selectbox(
                    "Ground‑truth label column",
                    options=list(df_gt.columns)
                )
                # merge on ID
                df_pred = df_pred.merge(
                    df_gt[[id_col, gt_col]],
                    on=id_col,
                    how="left",
                    validate="m:1"
                )
            elif gt_file and not id_col:
                st.error("Cannot merge ground‑truth file without an ID column.")
                gt_col = None
            else:
                gt_col = None

    # ───────────────────────────────────────────────────────
    #  3 ▸ SELECT TACTIC(S) FOR EVALUATION
    # ───────────────────────────────────────────────────────
    st.header("🎯 Choose tactics to evaluate")
    tactics_to_eval = st.multiselect(
        "Select one or more tactics",
        options=list(DEFAULT_TACTICS.keys()),
        default=list(DEFAULT_TACTICS.keys())[:1]  # pre‑select first tactic
    )

    # ───────────────────────────────────────────────────────
    #  4 ▸ RUN EVALUATION
    # ───────────────────────────────────────────────────────
    if st.button("🚀 Run Evaluation"):
        if not tactics_to_eval:
            st.error("Please select at least one tactic.")
            st.stop()

        # Helper to convert a cell to list safely
        def parse_list_cell(cell):
            if isinstance(cell, list):
                return cell
            try:
                return ast.literal_eval(cell)
            except Exception:
                return []

        df_pred["__pred_list__"] = df_pred[pred_col].apply(parse_list_cell)
        if gt_col:
            df_pred["__gt_list__"] = df_pred[gt_col].apply(parse_list_cell)

        # ▸ 4A Word‑level metrics aggregated at ID
        st.subheader("🧮 Word‑Level Metrics (aggregated by ID)")
        if id_col:
            rows = []
            for tactic in tactics_to_eval:
                key_terms = set(DEFAULT_TACTICS[tactic])
                # word counts per row
                df_pred["__total_words__"] = df_pred[text_col].str.split().apply(len)
                df_pred["__tactic_words__"] = df_pred[text_col].apply(
                    lambda txt: sum(1 for w in txt.split() if w in key_terms)
                )
                agg = df_pred.groupby(id_col)[["__total_words__", "__tactic_words__"]].sum()
                agg[f"{tactic}_pct_words"] = (
                    agg["__tactic_words__"] / agg["__total_words__"]
                ).fillna(0) * 100
                rows.append(agg[[f"{tactic}_pct_words"]])

            id_metrics = pd.concat(rows, axis=1).reset_index()
            st.dataframe(id_metrics.head())
            st.download_button(
                "📥 id_level_word_metrics.csv",
                id_metrics.to_csv(index=False).encode(),
                "id_level_word_metrics.csv",
                "text/csv"
            )
        else:
            st.info("No ID column selected → skipping ID‑level aggregation.")

        # ▸ 4B Classification metrics (precision, recall, F1)
        st.subheader("📊 Classification Metrics")

        if gt_col:
            metric_rows = []
            for tactic in tactics_to_eval:
                # flags per row
                df_pred["__pred_flag__"] = df_pred["__pred_list__"].apply(
                    lambda lst: tactic in lst
                )
                df_pred["__gt_flag__"] = df_pred["__gt_list__"].apply(
                    lambda lst: tactic in lst
                )

                TP = int(((df_pred["__pred_flag__"] == True) & (df_pred["__gt_flag__"] == True)).sum())
                FP = int(((df_pred["__pred_flag__"] == True) & (df_pred["__gt_flag__"] == False)).sum())
                FN = int(((df_pred["__pred_flag__"] == False) & (df_pred["__gt_flag__"] == True)).sum())

                precision = TP / (TP + FP) if (TP + FP) else 0.0
                recall    = TP / (TP + FN) if (TP + FN) else 0.0
                f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

                metric_rows.append({
                    "tactic": tactic,
                    "TP": TP, "FP": FP, "FN": FN,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })

            metrics_df = pd.DataFrame(metric_rows).set_index("tactic")
            st.dataframe(metrics_df.style.format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}"
            }))

            st.download_button(
                "📥 classification_metrics.csv",
                metrics_df.to_csv().encode(),
                "classification_metrics.csv",
                "text/csv"
            )
        else:
            st.info(
                "Ground‑truth labels were not provided; precision/recall/F1 cannot be computed."
            )
else:
    st.info("Upload the predictions CSV from the first app to begin.")
