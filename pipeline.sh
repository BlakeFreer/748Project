# The complete classification pipline.

# Usage: source pipeline.sh <partition>
# Where <partition> is the folder created by `partition.py`

out=$1
train=$1/train_data
test=$1/test_data

dimensions=12

if [ -f "$out/confusion.txt" ]; then
    echo "Error: $out/confusion.txt already exists. Exiting."
    exit 1
fi

echo "Extracting features from training data."
for f in $train/*.wav; do ./build/extract "$f" >> $out/train.txt; done

echo "Computing optimal basis."
./build/basis $out/train.txt > /dev/null

echo "Reducing dimensionality."
./build/reduce $out/train.txt $out/train $dimensions >> $out/train.reduced

echo "Training SVM."
./build/prep-svm $out/train.reduced
./third-party/libsvm/svm-train $out/train.svm $out/model > /dev/null

echo "Extracting features from test data."
for f in $test/*.wav; do ./build/extract "$f" >> $out/test.txt; done

echo "Projecting test data onto basis."
./build/reduce $out/test.txt $out/train $dimensions >> $out/test.reduced

echo "Predicting with SVM."
./build/prep-svm $out/test.reduced
./third-party/libsvm/svm-predict $out/test.svm $out/model $out/confusion.txt
