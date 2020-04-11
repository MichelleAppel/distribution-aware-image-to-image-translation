URL1=https://pjreddie.com/media/files/mnist_train.tar.gz
URL2=https://pjreddie.com/media/files/mnist_test.tar.gz

mkdir ./datasets/numbers
mkdir ./datasets/numbers/gz

gz_FILE1=./datasets/numbers/gz/mnist_train.tar.gz
gz_FILE2=./datasets/numbers/gz/mnist_test.tar.gz

path=./datasets/numbers/
train_path=./datasets/numbers/train
test_path=./datasets/numbers/test

if [ -f "$gz_FILE1" ]; then
    echo "$FILE exist"
else 
    wget -N $URL1 -O $gz_FILE1
fi

if [ -z "$(ls -A $train_path)" ]; then 
    tar -xvzf $gz_FILE1 -C $path
fi

if [ -f "$gz_FILE2" ]; then
    echo "$FILE exist"
else 
    wget -N $URL2 -O $gz_FILE2    
fi

if [ -z "$(ls -A $test_path)" ]; then 
    tar -xvzf $gz_FILE2 -C $path
fi

mv ./datasets/numbers/train ./datasets/numbers/trainB
mv ./datasets/numbers/test ./datasets/numbers/testB
