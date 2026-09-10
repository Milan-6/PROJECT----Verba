@echo off
cd /d %~dp0
echo Re-recording weak signs. Old samples are kept; new ones are added.
echo.
echo === BYE (was 0.63) ===
python collect_data.py --gloss BYE --signer bhargav --samples 25 --auto
echo.
echo === DAYS (was 0.57) ===
python collect_data.py --gloss DAYS --signer bhargav --samples 25 --auto
echo.
echo === HELLO (was 0.03) ===
python collect_data.py --gloss HELLO --signer bhargav --samples 25 --auto
echo.
echo === I_AM_FINE (was 0.83) ===
python collect_data.py --gloss I_AM_FINE --signer bhargav --samples 25 --auto
echo.
echo === NO (was 0.77) ===
python collect_data.py --gloss NO --signer bhargav --samples 25 --auto
echo.
echo === TWO (was 0.65) ===
python collect_data.py --gloss TWO --signer bhargav --samples 25 --auto
python train_mlp.py --epochs 40 --aug 4
python evaluate.py --holdout bhargav --min 0.85
pause
