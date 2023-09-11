ACCOUNT := noelmni
SERVICE := deep-fcd
IMAGE 	:= $(ACCOUNT)/$(SERVICE) # noelmni/deep-fcd
TAG		:= test
UID		:= 2551
GID		:= 618
CASE_ID := BAR_002
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
	/app/inference.py $(CASE_ID) t1.nii.gz flair.nii.gz /tmp cuda0 $(BRAIN_MASKING) $(PREPROCESS)

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