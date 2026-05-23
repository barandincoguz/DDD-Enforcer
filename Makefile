# WP-01b: paper-table generation
# Renders RQ1-4 LaTeX tables from runs/_aggregated/ into LaTeX_DL_468198_240419/tables/.

.PHONY: tables
tables:
	@echo "[wp-01b] aggregating run manifests..."
	cd extension/backend && python -m scripts.aggregate --runs-root runs/
	@echo "[wp-01b] rendering LaTeX tables..."
	cd extension/backend && python -m scripts.build_tables --runs-root runs/ --all --output-dir ../../LaTeX_DL_468198_240419/tables/
	@echo "[OK] tables generated under LaTeX_DL_468198_240419/tables/"
