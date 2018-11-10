
download:
	./download_and_convert_ade20k.sh

create-training-data:
	mkdir -p data/${LABEL_SET}
	python create_tfrecord_dataset.py \
		-i data/ADEChallengeData2016/images/training/ \
		-a data/ADEChallengeData2016/annotations/training/ \
		-o data/${LABEL_SET}/${LABEL_SET}_data.tfrecord \
		-l data/ADEChallengeData2016/objectInfo150.txt \
		-w "person, individual, someone, somebody, mortal, soul|house:building, edifice:house:skyscraper|sky|car, auto, automobile, machine, motorcar:bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle:truck, motortruck:van|bicycle, bike, wheel, cycle:minibike, motorbike" \
		-t 0.20

upload-data:
	gsutil cp data/${LABEL_SET}/* gs://${GCS_BUCKET}/data/${LABEL_SET}/


train-local-refine:
	python -m image_segmentation.train \
	    -d data/${LABEL_SET}/${LABEL_SET}_data.tfrecord \
	    -l data/${LABEL_SET}/labels.txt \
	    -n 10000 \
	    -s 768 \
	    -a 1 \
	    --steps-per-epoch 100 \
	    --batch-size 5 \
	    --lr 0.0001 \
            --fine-tune-checkpoint data/${LABEL_SET}/${LABEL_SET}_icnet_768x768_1_rf.h5 \
	    -o data/${LABEL_SET}/${LABEL_SET}_icnet_768x768_1_rf.h5 \
	    --refine

train-local:
	python -m image_segmentation.train \
	    -d data/${LABEL_SET}/${LABEL_SET}_data.tfrecord \
	    -l data/${LABEL_SET}/labels.txt \
	    -n 1000 \
	    -s 768 \
	    -a 1 \
	    --steps-per-epoch 50 \
	    --batch-size 32 \
	    --parallel-calls 8 \
	    --lr 0.0001 \
	    -o data/${LABEL_SET}/${LABEL_SET}_icnet_768x768_1.h5


train-cloud:
	python setup.py sdist
	gcloud ml-engine jobs submit training `whoami`_image_segmentation_`date +%s` \
	    --runtime-version 1.9 \
	    --job-dir=gs://${GCS_BUCKET} \
	    --packages dist/image_segmentation-1.0.tar.gz \
	    --module-name image_segmentation.train \
	    --region us-central1 \
	    --config config.yaml \
	    -- \
	    -d gs://fritz-data-sandbox/ADEChallengeData2016/people/people_data.tfrecord \
	    -l gs://fritz-data-sandbox/ADEChallengeData2016/people/labels.txt \
	    --fine-tune-checkpoint gs://fritz-image-segmentation-us-central/train/outdoor_objects_1536x1536_1_people_trained.h5 \
	    -o ${LABEL_SET}_1536x1536_1_people_trained_k80.h5 \
	    --image-size 1536 \
	    --alpha 1 \
	    --cores 1 \
	    --num-steps 2500 \
	    --batch-size 4 \
	    --gcs-bucket gs://${GCS_BUCKET}/train
