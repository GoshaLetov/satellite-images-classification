DROPBOX_URL := https://www.dropbox.com/scl/fi/7w4vlfcf6mfj8bstbgn1m/data.zip?rlkey=7sqlzv4pls6syb4zwigm7dgvd&dl=1
TEST_IMAGE_URL := https://www.dropbox.com/scl/fi/1eilp4x94mogvle6dhch8/train_0.jpg?rlkey=ux8i0djhaxvahlop5w8ilmw0z&dl=1

download_packages:
	apt-get update
	apt-get install unzip ffmpeg libsm6 libxext6 -y

download_train_data:
	wget --max-redirect=20 -O download.zip ${DROPBOX_URL}

download_test_data:
	wget --max-redirect=20 -O test.jpg ${TEST_IMAGE_URL}

preprocess_train_data:
	unzip download.zip
	rm -f download.zip
	rm -rf __MACOSX/

install_python_packages:
	pip install -U pip
	pip install -r requirements.txt

create_venv:
	python3.10 -m venv venv
	source venv/bin/activate
