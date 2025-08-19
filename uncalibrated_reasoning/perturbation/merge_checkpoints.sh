cd ../..

parent="$1"
for d in "$parent"*/; do
    [ -d "$d" ] || continue
    directory=${d%/}
    echo "Found directory: $directory"

    # Skip if actor_merged/ already exists
    if [ -d "$directory/actor_merged" ]; then
        echo "  → actor_merged/ exists – skipping merge"
        continue
    fi

    uv run python  verl/scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir $directory/actor/ \
        --target_dir $directory/actor_merged/
done

cd uncalibrated_reasoning/perturbation/
