#!/bin/bash

# Remote server details
REMOTE_USER="azureuser"
REMOTE_HOST="51.120.112.203"
REMOTE_DIR="app"
SSH_KEY="../Entrenando_key.pem"

#Lista de exclusiones
EXCLUDE=("ppo_robot_tensorboard/jump" "ppo_robot_tensorboard/walk" ".git" ".gitignore" "models/jump" "models/walk")

# Crear una lista de archivos y carpetas a subir, excluyendo los que están en la lista
FILES_TO_UPLOAD=()
for file in *; do
    # Comprobar si el archivo o directorio está en la lista de exclusión
    skip=false
    for exclude in "${EXCLUDE[@]}"; do
        if [[ "$file" == "$exclude" ]]; then
            skip=true
            break
        fi
    done
    # Si no está en la lista de exclusión, agregarlo a la lista de archivos para subir
    if [[ "$skip" == false ]]; then
        FILES_TO_UPLOAD+=("$file")
    fi
done

# Subir archivos usando sftp
sftp -i "${SSH_KEY}" ${REMOTE_USER}@${REMOTE_HOST} << EOF
cd ${REMOTE_DIR}
$(for file in "${FILES_TO_UPLOAD[@]}"; do
    if [[ -d "$file" ]]; then
        echo "mkdir -p $file"
        echo "put -r \"$file\" \"$file\""
    else
        echo "put \"$file\" \"$file\""
    fi
done)
bye
EOF

echo "Subida de archivos completada."
