# ───────────────────────────────────────────────────────────
#  tactic_evaluator_app.py   (robust version)
#  ----------------------------------------------------------
#  Evaluates output from “Marketing‑Tactic Text Classifier”
#  • ID‑level word‑hit metrics
#  • Precision / Recall / F1
#  ----------------------------------------------------------
import ast
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
    #  2 ▸ COLUMN MAPPING
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

    # Ground‑truth labels
    with st.expander("Add ground‑truth labels (optional)"):
        has_gt = st.checkbox("Ground‑truth labels are in the same file", value=False)
        if has_gt:
            gt_col = st.selectbox(
                "Column containing *true* categories list",
                options=list(df_pred.columns)
            )
        else:
            gt_file = st.file_uploader(
                "📁 Upload a separate CSV with ground‑truth labels (must share ID)",
                type="csv",
                key="gt_csv"
            )
            if gt_file and id_col:
                df_gt = pd.read_csv(gt_file)
                st.success(f"Loaded {len(df_gt):,} ground‑truth rows.")
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
            else:
                gt_col = None

    # ───────────────────────────────────────────────────────
    #  3 ▸ SELECT TACTICS
    # ───────────────────────────────────────────────────────
    st.header("🎯 Choose tactics to evaluate")
    tactics_to_eval = st.multiselect(
        "Select one or more tactics",
        options=list(DEFAULT_TACTICS.keys()),
        default=list(DEFAULT_TACTICS.keys())[:1]
    )

    # ───────────────────────────────────────────────────────
    #  4 ▸ RUN EVALUATION
    # ───────────────────────────────────────────────────────
    if st.button("🚀 Run Evaluation"):
        if not tactics_to_eval:
            st.error("Please select at least one tactic.")
            st.stop()

        # ---------- Safe tokeniser ----------
        def tokens(x):
            return x.split() if isinstance(x, str) else []

        # Build helper columns once
        df_pred["__tokens__"]      = df_pred[text_col].apply(tokens)
        df_pred["__total_words__"] = df_pred["__tokens__"].apply(len)

        # ▸ 4A Word‑level metrics aggregated at ID
        st.subheader("🧮 Word‑Level Metrics (aggregated by ID)")
        if id_col:
            agg_frames = []
            for tactic in tactics_to_eval:
                key_terms = set(DEFAULT_TACTICS[tactic])

                df_pred["__tactic_words__"] = df_pred["__tokens__"].apply(
                    lambda tok: sum(1 for w in tok if w in key_terms)
                )

                agg = df_pred.groupby(id_col)[["__total_words__", "__tactic_words__"]].sum()
                agg[f"{tactic}_pct_words"] = (
                    (agg["__tactic_words__"] / agg["__total_words__"]).fillna(0) * 100
                )
                agg_frames.append(agg[[f"{tactic}_pct_words"]])

            id_metrics = pd.concat(agg_frames, axis=1).reset_index()
            st.dataframe(id_metrics.head())
            st.download_button(
                "📥 id_level_word_metrics.csv",
                id_metrics.to_csv(index=False).encode(),
                "id_level_word_metrics.csv",
                "text/csv"
            )
        else:
            st.info("No ID column selected → skipping ID‑level aggregation.")

        # ▸ 4B Classification metrics
        st.subheader("📊 Classification Metrics")
        if gt_col:
            # Helper: parse list-like cells
            def to_list(cell):
                if isinstance(cell, list):        return cell
                if isinstance(cell, str) and cell.startswith("["):  # likely repr
                    try: return ast.literal_eval(cell)
                    except Exception: pass
                return []

            df_pred["__pred_list__"] = df_pred[pred_col].apply(to_list)
            df_pred["__gt_list__"]   = df_pred[gt_col].apply(to_list)

            records = []
            for tactic in tactics_to_eval:
                df_pred["__pred_flag__"] = df_pred["__pred_list__"].apply(lambda lst: tactic in lst)
                df_pred["__gt_flag__"]   = df_pred["__gt_list__"].apply(lambda lst: tactic in lst)

                TP = int(((df_pred["__pred_flag__"]) & (df_pred["__gt_flag__"])).sum())
                FP = int(((df_pred["__pred_flag__"]) & (~df_pred["__gt_flag__"])).sum())
                FN = int((~df_pred["__pred_flag__"] & (df_pred["__gt_flag__"])).sum())

                precision = TP / (TP + FP) if (TP + FP) else 0.0
                recall    = TP / (TP + FN) if (TP + FN) else 0.0
                f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

                records.append({
                    "tactic": tactic,
                    "TP": TP, "FP": FP, "FN": FN,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })

            metrics_df = pd.DataFrame(records).set_index("tactic")
            st.dataframe(
                metrics_df.style.format({"precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"})
            )
            st.download_button(
                "📥 classification_metrics.csv",
                metrics_df.to_csv().encode(),
                "classification_metrics.csv",
                "text/csv"
            )
        else:
            st.info("Ground‑truth labels not provided → precision/recall/F1 not computed.")
else:
    st.info("Upload the predictions CSV from the first app to begin.")
