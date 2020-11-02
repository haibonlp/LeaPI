

ep=200
none=300
lr=1e-3
seed=111
sampletimes=30

init=default
wvfpath=../resources/GoogleNews-vectors-negative300.bin



echo python main_learn.py --rl-max-epoch $ep --rl-sample-times $sampletimes --rl-agent-lr  $lr  --state-feat prodlabel   --none-size $none  --wvfpath $wvfpath  --seed $seed --rl-init $init 



