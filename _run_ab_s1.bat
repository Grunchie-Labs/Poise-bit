@echo off
cd /d "D:\College\Ongoing Projects\Grunchie Labs\Poise-bit"
python research\experiments\ablation.py --eval-seed 1 --out results\ablation_seed1_v2.json > out_ab_s1d.log 2>&1
echo DONE_S1 >> out_ab_s1d.log
