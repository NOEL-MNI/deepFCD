ACCOUNT := noelmni
SERVICE := deep-fcd
IMAGE 	:= $(ACCOUNT)/$(SERVICE) # noelmni/deep-fcd
TAG		:= test
UID		:= 2551 # id -u
GID		:= 618 # id -g
CASE_ID := mcd_0468_1
TMPDIR	:= /host/hamlet/local_raid/data/ravnoor/sandbox/testrun

build:
	docker build -t $(ACCOUNT)/$(SERVICE):$(TAG) .

clean-build:
	docker build -t $(ACCOUNT)/$(SERVICE):$(TAG) . --no-cache

test-pipeline:
	./app/inference.py $(CASE_ID) t1.nii.gz flair.nii.gz $(TMPDIR) cuda0 1 1

test-preprocess:
	./app/preprocess.sh $(CASE_ID) t1.nii.gz flair.nii.gz $(TMPDIR) 1 1

test-pipeline-docker:
	docker run --rm -it --init \
	--gpus=all	\
	--user="$(UID):$(GID)" \
	--volume="$(TMPDIR):/tmp" \
	$(ACCOUNT)/$(SERVICE):$(TAG) \
	/app/inference.py $(CASE_ID) t1.nii.gz flair.nii.gz /tmp cuda0 1 1

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