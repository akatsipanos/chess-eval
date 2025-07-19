# TODO

- Tuning class, this does not work currently. Main blocker is figuring out what to do with params. Initially I though this would be user input but I forgot the suggest_xxx function comes from the trial variable, specified in the objective function. Need to figure out if there's a workaround here

- Get rid of the argparse in the ml modelling - so much better with a config file, can see previous runs instantly and easilly replicate

- API - improve plus make the frontend interactive - interactive chess board with live updating ML prediction with changing position.

- Try something other than mypy, takes far too long

- Work on improving the ML model, this has not been touched at all after the 2 year hiatus.

- Write tests

- Clean up - remove old tuning files if tuning class is a success; remove flask apps once completed the interactice fastapi-js one.

- Add in pyrefly - mypy is too slow