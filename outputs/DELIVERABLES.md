# Deliverables

Final full-data artifacts:

- Full Chinese presentation: `outputs/LLM_sycophancy_locate_steer_improve_presentation_full.pptx`
- Full Chinese report: `outputs/report_zh_full.md`
- Full experiment input: `outputs/qwen_mmlu_raw_full.csv`
- Full experiment outputs: `outputs/qwen_mmlu_raw_full_locate_steer_improve/`
- Full-result visualizations and summary tables: `outputs/full_analysis/`
- Full deck contact sheet: `outputs/final-contact-sheet-full.png`
- Code: `outputs/code/run_qwen3000_cpu_suite.py`, `scripts/build_mmlu_raw_full_csv.py`, `scripts/remote_seeta_job.py`, `scripts/summarize_full_results.py`, `scripts/build_full_sycophancy_presentation.mjs`

Full experiment coverage:

- Behavior locate: `locate_behavior.csv`, `locate_behavior_summary.csv`
- Logit Lens locate: `locate_layer_logit_lens.csv`, `locate_layer_summary.csv`
- Activation patching locate: `activation_patching_summary.csv`, `activation_patching_layer_summary.csv`
- Vector steering: `steering_vector*.pt`, `steering_vectors.pt`, `steer_sweep.csv`, `steer_sweep_summary.csv`
- Prompt improve: `improve_prompt_mitigation.csv`, `improve_prompt_mitigation_summary.csv`
- Elimination analysis: `sycophancy_elimination_summary.csv`
- Run metadata: `run_config.json`

Verified full scale:

- Source data: `D:\26_ML_MIProject\raw_data\mmlu_raw.pkl`
- Experiment CSV rows: 14042
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Device: AutoDL/SeetaCloud CUDA
- Behavior rows: 42126
- Logit Lens rows: 1011024
- Steering sweep rows: 393176
- Prompt mitigation rows: 70210
- Activation patching rows: 337008

Key full-data findings:

- Opinion-only prompting raises sycophancy rate from 19.1% to 69.2%.
- Logit Lens finds the strongest opinion-over-answer signal around layers 20-22, peaking at layer 21.
- Activation patching gives the strongest causal recovery at layer 22, followed by layers 21 and 20.
- Simple mean-vector steering is directionally consistent but weak; the best full sweep elimination rate is 6.4%.
- `anti_sycophancy` is the strongest prompt mitigation, reducing overall sycophancy rate to 39.3% and raising accuracy to 30.3%.

Template provenance:

- User template: `inputs/智科院紫.pptx`
- ASCII working copy: `inputs/template_zky_purple.pptx`
- Full deck QA renders: `outputs/LLM_sycophancy_locate_steer_improve_presentation_full/`

Qwen2.5-1.5B full-data addendum:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Full experiment outputs: `outputs/qwen15_mmlu_raw_full_locate_steer_improve/`
- New Chinese report: `outputs/report_zh_full_qwen15.md`
- New Chinese presentation: `outputs/LLM_sycophancy_locate_steer_improve_presentation_qwen15.pptx`
- New deck contact sheet: `outputs/final-contact-sheet-qwen15.png`
- New visualizations and summary tables: `outputs/full_analysis_qwen15/`
- Verified rows: behavior 42126, logit lens 1179528, steering 393176, prompt mitigation 70210, activation patching 393176.
- Key result: opinion-only sycophancy is 32.5%; best prompt mitigation is `anti_sycophancy` at 27.4% overall sycophancy; best steering elimination is 0.9%.
