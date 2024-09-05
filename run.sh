for i in {0..1}
do
    bash _5_train.sh $i > _5_train_$i.log 2>&1 &
done

bash _5_train.sh 0 > _5_train_0.log 2>&1 &
