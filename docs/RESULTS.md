# Where every number comes from

No number in the paper is typed by hand. Every value that depends on data or runs is a LaTeX macro `\res{key}`;
`fin fill-results` computes all keys from the committed run outputs and writes them, one `\resdef{key}{value}` per
line, to `sections/results_values.tex` of the manuscript folder (`--paper-dir` or `FIN_PAPER`); it also writes the
tables in `sections/tables/`. The manuscript registers every key it uses in `RESULT_KEYS.md`
(`key | meaning | source`); a meaning such as `(%, 1 dp)` or `mean over seeds` sets the unit, precision or part of
that key.

The README results tables (`fin readme-table`) and charts (`fin repo-figures`) are computed by
`fin.reporting.summary` directly from `results/runs` and `splits/frames.csv`, with the same definitions as the
corresponding paper values: classification mean ± sample SD over folds (cross-validation) or seeds (locked test
split); segmentation per-image Dice and AoP MAE, mean ± sample SD over seeds; the constant AoP prediction is the
mean reference AoP of the training SPHead frames; the ground-truth-mask floor is the AoP construction applied to
the ground-truth masks.

## Sources

All files under `results/runs/` are copied from `$FIN_RUNS` by `fin collect-results` (the `report` stage), which
writes local working-folder paths as `$FIN_WORK/...`.

| Run outputs (`results/runs/`) | Written by | Used for |
|---|---|---|
| `cls/<model>/seed<k>/{metrics_test.json, preds_test.npz, preds_val.npz, config.json, done.json, epochs.jsonl}` | `fin train-cls`, `fin eval-cls` | fixed-split classification table (supplement), gate analysis, ROC figure, paired differences |
| `cls/summary.json` | `fin aggregate cls` | Youden table, split check |
| `cls_cv/fold<k>/<model>/seed0/...`, `cls_cv/summary_cv.json` | `fin schedule cv`, `fin aggregate cv` | Table 1, cross-validation (primary classification comparison) |
| `cls_authorsplit/<model>/seed<k>/...`, `author_split_check.json` | `fin schedule authorsplit`, `fin aggregate author-split` | sensitivity of the classification result to the split (supplement) |
| `seg/<model>/seed<k>/{metrics.json, per_image_{jnu_test,iugc}.csv, preds_*.npz, log.jsonl}`, `seg/ceiling.json`, `seg/summary.json` | `fin train-seg`, `fin eval-seg`, `fin aggregate seg` | Tables 4 and 5, Figures 7 and 8, Bland-Altman analysis and pooled table (supplement) |
| `ext469/{summary.json, per_image_all.csv, <model>_seed<k>.csv}` | `fin ext469` | second external AoP set (supplement, abstract) |
| `robustness/<model>/seed<k>/{metrics.json, preds.npz}`, `robustness/summary.json` | `fin robustness` | Table 3, supplementary AUC table |
| `efficiency/results.json` | `fin efficiency` | Table 2 and the complexity columns of Table 4 |
| `e2e/summary.json` | `fin e2e` | same statistics as the `aop.e2e.*` keys (not read by `fin fill-results`) |
| `splits/frames.csv`, `splits/cv/folds.json` | `fin jnu build`, `fin match-aop`, `fin splits cv` | dataset counts, constant AoP baseline, fold sizes |
| `$FIN_DATA/iugc/processed/aop.csv`, `$FIN_SEG_CACHE` | `fin build-iugc`, `fin cache seg` | IUGC 2024 references, pixel accuracy, image-blind reference rows |

The end-to-end values (`aop.e2e.*`) are computed from the classification predictions and the segmentation
per-frame CSVs (`fin.evaluation.e2e`; `fin e2e` writes the same statistics as JSON).

## Key scheme

Plain key = `mean ± SD` over seeds; `.mean` / `.sd` = one part; `.ci` = 95% CI `[lo, hi]`.

| Prefix | Keys |
|---|---|
| `data.` | `data.jnu.{frames,videos,women,none,onlysp,onlyhead,sphead,standard,nonstandard,mismatch,width,height}`, `data.split.<train/val/test>.{frames,videos,women,standard,nonstandard,<category>,seg.frames,aop.frames,aop.mean,aop.sd}`, `data.iugc.{frames,aop.mean,aop.sd,refdiff.mae,refdiff.p95}`, `data.jnu.refdiff.{mae,p95,max}`, `data.jnu.<category>.<fh/ps>_px_pct` |
| `cls.` | `cls.<model>.<acc/prec/sens/spec/f1/mf1/auc>[.mean/.sd/.ci]` (Standard positive, threshold 0.5, test), `cls.<model>.youden.<thr/acc/sens/spec/mf1>`, `cls.<model>.val.<auc/mf1>`, `cls.<model>.gate.<forwarded/purity/leak/missed/leak.<cat>/leakrate.<cat>>`, `cls.<model>.vsdfn.<acc/mf1/auc>[.ci/.excl0]` (paired video bootstrap, model minus DeepFetalNet), `cls.rank.<metric>.first`, `cls.baseline.valbest`, `cls.{ntest,ntest.standard,ntest.nonstandard,ntestvideos,nboot,seeds}`, `cls.best.<test/val>.<metric>.<name/mean>` |
| `cv.` | `cv.<model>.<acc/mf1/auc/sens/spec>[.mean/.sd/.range]`, `cv.<model>.pooled.<mf1/auc/sens/spec>`, `cv.<model>.rank.<mf1/auc>[.best/.worst/.mean]`, `cv.<model>.vsdfn.<mf1/auc>[.ci/.excl0]`, `cv.<model>.vsdfn.foldsbetter`, `cv.{k,nboot,val.frames}`, `cv.test.<women/frames>.<min/max>`, `cv.nsigbetter.<mf1/auc>`, `cv.best.mf1.<name/mean>` |
| `asplit.` | `asplit.<model>.<val/all/seen/unseen/patient_split_test>.<acc/mf1/auc>[.mean]`, `asplit.<model>.<set>.n` |
| `eff.` | `eff.<model>.{params,params.m,gmacs,gmacs.thop,gmacs.fvcore,lat.<mps/cpu>[.iqr/.mean/.sd],thr.<mps/cpu>,mem.weights,input}`, `eff.hw.{cpu,torch,macos,ncpu,memgb}` |
| `seg.` | `seg.<model>.<jnu/iugc>.<dice.fh,dice.ps,meandice,iou.fh,iou.ps,meaniou,hd95.fh,hd95.ps,hd95n.fh,hd95n.ps,bf1.fh,bf1.ps,pdice.*,piou.*,pixacc,empty.*,hdundef.*>`, `seg.<model>.{seeds,nseeds,params,params.m}`, reference rows `seg.<prior/bg>.<set>.<metric>`, `seg.nframes.<set>` |
| `aop.` | `aop.<src>.<jnu/iugc>.{n,nfail,succ,mae,maeci,rmse,medae,bias,sd,loalow,loahigh,loa,r,w5,w10,w5fi,w10fi,maxae,sec.mae}` with `src` = `unet, fatnet, malunet, aopgaunet` (geometry), `aopgaunet.head` (regression head), `gt` (geometry on the ground-truth masks), `const` (training-mean AoP); `aop.const.value`; end to end `aop.e2e.jnu.*`, `aop.e2e.{gate,segmenter,seeds}`; `aop.<set>.<metric>.<min/max>[.name]` over the four segmenters |
| `rob.` | `rob.<model>.<condition>.{acc,sens,spec,auc,mf1,dauc}` with condition in `clean, speckle-0.05/0.10/0.20, blur-5/9/15, gauss-0.01/0.05/0.10`; `rob.nframes` |
| `ext469.` | `ext469.n.{all,labelled,landmarkonly,unseen,unseen.<iugctest/iugcval/jnutest>,jnu,jnu.<train/val/test/women/videos>,<iugctest/iugctrain/iugcval>.labelled}`, `ext469.refdiff.max`, `ext469.const.value`, `ext469.<const/gt>.<all/labelled/unseen>.mae` (+ `gt.*.n`), `ext469.<model>[.head].<all/labelled/unseen>.mae[.mean/.sd/.seeds]`, `ext469.<model>.nseeds`, `ext469.nfail.max` |

## Statistical conventions

- Classification: Standard (SPHead) positive, threshold 0.5, mean ± sample SD over seeds; the 95% CI is the
  percentile cluster bootstrap over the test videos of the seed-averaged metric (2000 resamples, the same video draw
  for every seed); paired differences to DeepFetalNet use the same bootstrap. Cross-validation: mean ± SD over folds;
  the pooled difference to DeepFetalNet has a CI from a bootstrap over the 51 women.
- Segmentation: per-image mean over the test frames, then mean ± SD over seeds; pooled Dice/IoU in the supplement.
- AoP: error statistics over frames with a valid angle; failures are counted and, in the failure-inclusive
  analysis, treated as misses; MAE CI by cluster bootstrap over videos (frames for IUGC 2024, one per video).

## Checking a number

1. Find the key in the manuscript source (`\res{...}`) and its line in the manuscript's `RESULT_KEYS.md`.
2. `fin fill-results --paper-dir DIR --out-dir OUT --dump OUT/keys.json` writes every key and value as JSON.
3. The source column of the registry names the run output; `fin.reporting.values` holds the computation.
