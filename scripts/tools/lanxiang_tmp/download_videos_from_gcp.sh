BUCKET="science_compass"
BASE_PATH="evaluation_videos"
USER_NAME="daniel"

# Create a local directory for downloads if it doesn't exist
mkdir -p judge/data/evaluation_videos

# Loop through all model folders
for MODEL_PATH in $(gsutil ls gs://${BUCKET}/${BASE_PATH}/ | grep -v "gs://${BUCKET}/${BASE_PATH}/$"); do
  MODEL_NAME=$(basename ${MODEL_PATH})
  USER_PATH=${MODEL_PATH}${USER_NAME}
  DEST_DIR=judge/data/evaluation_videos/${MODEL_NAME}
  echo "=== Checking ${USER_PATH} ==="

  # Check if daniel folder exists, then download
  if gsutil ls ${USER_PATH} >/dev/null 2>&1; then
    echo "--> Downloading from ${MODEL_NAME}/${USER_NAME} ..."
    mkdir -p "${DEST_DIR}"
    gsutil -m cp -r "${USER_PATH}" "${DEST_DIR}"
  else
    echo "WARNING!!! No ${USER_NAME}/ folder in ${MODEL_NAME}, skipping."
  fi
done
