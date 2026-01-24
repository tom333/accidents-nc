#!/bin/bash
# Script de téléchargement des données d'accidents depuis data.gouv.fr
# Usage: ./download_data.sh

set -e

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}📥 TÉLÉCHARGEMENT DES DONNÉES D'ACCIDENTS${NC}\n"

mkdir -p data

# URLs des fichiers caractéristiques
declare -a URLS_CARACTERISTIQUES=(
    "https://www.data.gouv.fr/fr/datasets/r/e22ba475-45a3-46ac-a0f7-9ca9ed1e283a"
    "https://www.data.gouv.fr/fr/datasets/r/07a88205-83c1-4123-a993-cba5331e8ae0"
    "https://www.data.gouv.fr/fr/datasets/r/85cfdc0c-23e4-4674-9bcd-79a970d7269b"
    "https://www.data.gouv.fr/fr/datasets/r/5fc299c0-4598-4c29-b74c-6a67b0cc27e7"
    "https://www.data.gouv.fr/fr/datasets/r/104dbb32-704f-4e99-a71e-43563cb604f2"
    "https://www.data.gouv.fr/api/1/datasets/r/83f0fb0e-e0ef-47fe-93dd-9aaee851674a"
)

declare -a FILENAMES_CARACTERISTIQUES=(
    "caracteristiques-2024.csv"
    "caracteristiques-2023.csv"
    "caracteristiques-2022.csv"
    "caracteristiques-2021.csv"
    "caracteristiques-2020.csv"
    "caracteristiques-2019.csv"
)

# URLs des fichiers usagers
declare -a URLS_USAGERS=(
    "https://www.data.gouv.fr/fr/datasets/r/36b1b7b3-84b4-4901-9163-59ae8a9e3028"
    "https://www.data.gouv.fr/fr/datasets/r/78c45763-d170-4d51-a881-e3147802d7ee"
    "https://www.data.gouv.fr/fr/datasets/r/ba5a1956-7e82-41b7-a602-89d7dd484d7a"
    "https://www.data.gouv.fr/fr/datasets/r/62c20524-d442-46f5-bfd8-982c59763ec8"
    "https://www.data.gouv.fr/fr/datasets/r/68848e2a-28dd-4efc-9d5f-d512f7dbe66f"
    "https://www.data.gouv.fr/api/1/datasets/r/f57b1f58-386d-4048-8f78-2ebe435df868"
)

declare -a FILENAMES_USAGERS=(
    "usagers-2024.csv"
    "usagers-2023.csv"
    "usagers-2022.csv"
    "usagers-2021.csv"
    "usagers-2020.csv"
    "usagers-2019.csv"
)

download_file() {
    local url=$1
    local filename=$2
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        echo -e "${YELLOW}⬇️  Téléchargement: ${filename}${NC}"
        
        if curl -L --fail --silent --show-error --output "data/${filename}" "${url}"; then
            local size=$(du -h "data/${filename}" | cut -f1)
            echo -e "${GREEN}✅ ${filename} téléchargé (${size})${NC}"
            return 0
        else
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo -e "${YELLOW}⚠️  Échec, nouvelle tentative (${retry}/${max_retries})...${NC}"
                sleep 2
            fi
        fi
    done
    
    echo -e "${RED}❌ Échec du téléchargement de ${filename}${NC}"
    return 1
}

echo -e "\n${BLUE}📋 Fichiers Caractéristiques (accidents)${NC}"
success_count_carac=0
for i in "${!URLS_CARACTERISTIQUES[@]}"; do
    url="${URLS_CARACTERISTIQUES[$i]}"
    filename="${FILENAMES_CARACTERISTIQUES[$i]}"
    if download_file "$url" "$filename"; then
        success_count_carac=$((success_count_carac + 1))
    fi
done

echo -e "\n${BLUE}📋 Fichiers Usagers (victimes)${NC}"
success_count_usagers=0
for i in "${!URLS_USAGERS[@]}"; do
    url="${URLS_USAGERS[$i]}"
    filename="${FILENAMES_USAGERS[$i]}"
    if download_file "$url" "$filename"; then
        success_count_usagers=$((success_count_usagers + 1))
    fi
done

total_success=$((success_count_carac + success_count_usagers))
total_files=$((${#URLS_CARACTERISTIQUES[@]} + ${#URLS_USAGERS[@]}))

echo -e "\n${BLUE}📊 RÉCAPITULATIF${NC}"
echo "----------------------------------------"
echo -e "Fichiers téléchargés: ${GREEN}${total_success}/${total_files}${NC}"
echo -e "  - Caractéristiques: ${success_count_carac}/6"
echo -e "  - Usagers:          ${success_count_usagers}/6"
total_size=$(du -sh data 2>/dev/null | cut -f1)
echo -e "Espace disque:        ${GREEN}${total_size}${NC}"

if [ $total_success -ge 10 ]; then
    echo -e "\n${GREEN}✅ TÉLÉCHARGEMENT RÉUSSI${NC}"
    echo -e "\nProchaines étapes:"
    echo -e "  1. ${YELLOW}python precompute_density.py${NC}"
    echo -e "  2. ${YELLOW}marimo edit accident_fetch_data.py${NC}"
    exit 0
else
    echo -e "\n${RED}⚠️  TÉLÉCHARGEMENT INCOMPLET${NC}"
    echo -e "Au moins 10 fichiers requis (sur 12)"
    exit 1
fi
