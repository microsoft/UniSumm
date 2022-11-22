### train all sub-tasks in SummZoo in both setting using 4 GPUs

for seed in {42..46}
do
    for task in dialogsum samsum arxiv reddit qmsum multinews wikihow
    do
        train.sh universal ${task} ${seed} 100 100 1000
        train.sh universal ${task} ${seed} 10 10 100
    done
done

for seed in {42..46}
do
    train.sh universal xsum ${seed} 100 10 100
    train.sh universal xsum ${seed} 10 1 10
done



### test all sub-tasks using 2 GPUs

mkdir ../unisumm_outs

for seed in {42..46}
do
    bash test.sh universal samsum  ${seed}  128 15 0 100 1000 &
    bash test.sh universal samsum  ${seed}  128 15 1 10 100
done

for seed in {42..46}
do
    bash test.sh universal dialogsum  ${seed}  100 15 0 100 1000 &
    bash test.sh universal dialogsum  ${seed}  100 15 1 10 100
done

for seed in {42..46}
do
    bash test.sh universal arxiv  ${seed}  256 128 0 100 1000 &
    bash test.sh universal arxiv  ${seed}  256 128 1 10 100
done

for seed in {42..46}
do
    bash test.sh universal reddit  ${seed}  128 30 0 100 1000 &
    bash test.sh universal reddit  ${seed}  128 30 1 10 100
done

for seed in {42..46}
do
    bash test.sh universal qmsum  ${seed}  256 60 0 100 1000 &
    bash test.sh universal qmsum  ${seed}  256 60 1 10 100
done

for seed in {42..46}
do
    bash test.sh universal multinews  ${seed} 400 256 0 100 1000 &
    bash test.sh universal multinews  ${seed} 400 256 1 10 100
done

for seed in {42..46}
do
    bash test.sh universal wikihow  ${seed} 256 30 0 100 1000 &
    bash test.sh universal wikihow  ${seed} 256 30 1 10 100
done

for seed in {42..46}
do
    bash test.sh universal xsum  ${seed} 64 15 0 100 1000 &
    bash test.sh universal xsum  ${seed} 64 15 1 10 100
done
