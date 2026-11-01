@echo off
cd /d "D:\College\Ongoing Projects\Grunchie Labs\Poise-bit"
python research\experiments\ablation.py > out_ab_s0d.log 2>&1
echo DONE_S0 >> out_ab_s0d.log
