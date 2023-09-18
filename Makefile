ACCOUNT := noelmni
SERVICE := deep-fcd
IMAGE 	:= $(ACCOUNT)/$(SERVICE) # noelmni/deep-fcd
TAG		:= resamp_orig
UID		:= 2551
GID		:= 618
CASE_ID := sub-00055
TMPDIR	:= /host/hamlet/local_raid/data/ravnoor/sandbox
BRAIN_MASKING := 1
PREPROCESS		:= 1

.PHONY: all clean

build:
	docker build -t $(ACCOUNT)/$(SERVICE):$(TAG) .

clean-build:
	docker build -t $(ACCOUNT)/$(SERVICE):$(TAG) . --no-cache

test-pipeline:
	./app/inference.py $(CASE_ID) t1.nii.gz flair.nii.gz $(TMPDIR) cuda0 $(BRAIN_MASKING) $(PREPROCESS)

memray-profiling:
	python3 -m memray run ./app/inference.py $(CASE_ID) t1_brain.nii.gz t2_brain.nii.gz $(TMPDIR) cuda0 0 0

memray-profiling-cpu:
	python3 -m memray run ./app/inference.py $(CASE_ID) t1_brain.nii.gz t2_brain.nii.gz $(TMPDIR) cpu 0 0

test-preprocess:
	./app/preprocess.sh $(CASE_ID) t1.nii.gz flair.nii.gz $(TMPDIR) $(BRAIN_MASKING) $(PREPROCESS)

test-pipeline-docker:
	docker run --rm -it --init \
	--gpus=all	\
	--user="$(UID):$(GID)" \
	--volume="$(TMPDIR):/tmp" \
	$(ACCOUNT)/$(SERVICE):$(TAG) \
	/app/inference.py $(CASE_ID) T1.nii.gz FLAIR.nii.gz /tmp cuda0 $(BRAIN_MASKING) $(PREPROCESS)

test-pipeline-docker_ci:
	docker run --rm -it --init \
	--gpus=all	\
	--user="$(UID):$(GID)" \
	--volume="$(TMPDIR):/tmp" \
	--env CI_TESTING=1 \
	--env CI_TESTING_GT=/tmp/$(CASE_ID)/label_final_MD.nii.gz \
	$(ACCOUNT)/$(SERVICE):$(TAG) \
	/app/inference.py $(CASE_ID) T1.nii.gz FLAIR.nii.gz /tmp cuda0 $(BRAIN_MASKING) $(PREPROCESS)

test-reporting:
	./app/utils/reporting.py $(CASE_ID) $(TMPDIR)/

install-jupyter-kernel:
	python -m ipykernel install --user --name deepFCD

clean:
	rm -rf $(TMPDIR)/$(CASE_ID)/{tmp,native,transforms}
	rm -f $(TMPDIR)/$(CASE_ID)/{*_final,*denseCrf3d*,*_native,*_maskpred}.nii.gz

docker-clean:
	docker run --rm -it --init \
	--volume="$(TMPDIR):/tmp" \
	busybox:latest \
	rm -rf /tmp/$(CASE_ID)/{tmp,native,transforms,noel_deepFCD_dropoutMC} && \
	rm -f /tmp/$(CASE_ID)/{*_final,*denseCrf3d*,*_native,*_maskpred}.nii.gz

prune:
	docker image prune

runner-build:
	docker-compose -f runner.docker-compose.yml build

runner-ps:
	docker-compose -f runner.docker-compose.yml ps

runner-up:
	docker-compose -f runner.docker-compose.yml up --remove-orphans -d

runner-down:
	docker-compose -f runner.docker-compose.yml down

runner-logs:
	docker-compose -f runner.docker-compose.yml logs -f

runner-scale:
	docker-compose up --scale runner=1 -d

runner-bash:
	docker-compose -f runner.docker-compose.yml exec -it runner bash