URL1=http://ufldl.stanford.edu/housenumbers/train_32x32.mat
URL2=http://ufldl.stanford.edu/housenumbers/test_32x32.mat
URL3=http://ufldl.stanford.edu/housenumbers/extra_32x32.mat

mkdir ./datasets/numbers
mkdir ./datasets/numbers/mat
mkdir ./datasets/numbers/trainA
mkdir ./datasets/numbers/testA

mat_FILE1=./datasets/numbers/mat/train_32x32.mat
mat_FILE2=./datasets/numbers/mat/test_32x32.mat
mat_FILE3=./datasets/numbers/mat/extra_32x32.mat

if [ -f "$mat_FILE1" ]; then
    echo "$FILE exist"
else 
    wget -N $URL1 -O $mat_FILE1
fi

if [ -f "$mat_FILE2" ]; then
    echo "$FILE exist"
else 
    wget -N $URL2 -O $mat_FILE2
fi

if [ -f "$mat_FILE3" ]; then
    echo "$FILE exist"
else 
    wget -N $URL3 -O $mat_FILE3
fi

files_train=./datasets/numbers/trainA/
if [ -z "$(ls -A $files_train)" ]; then 
    python3 ./datasets/Preprocess-SVHN/src/svhn.py --dataset train --input_mat datasets/numbers/mat/train_32x32.mat --output_path datasets/numbers/trainA
    python3 ./datasets/Preprocess-SVHN/src/svhn.py --dataset train --input_mat datasets/numbers/mat/extra_32x32.mat --output_path datasets/numbers/trainA
fi

files_test=./datasets/numbers/testA/
if [ -z "$(ls -A $files_test)" ]; then 
    python3 ./datasets/Preprocess-SVHN/src/svhn.py --dataset test --input_mat datasets/numbers/mat/test_32x32.mat --output_path datasets/numbers/testA
fi

# rm ./datasets/numbers/mat -r