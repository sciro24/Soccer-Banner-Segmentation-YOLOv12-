# Soccer Banner Segmentation Pipeline

Questo progetto implementa una pipeline completa orientata alla computer vision per l'individuazione e l'elaborazione di cartelloni pubblicitari all'interno di riprese di partite di calcio. L'obiettivo primario e quello di mascherare o sostituire le tele pubblicitarie a bordo campo, applicando un effetto di "green screen" in post-produzione, isolando in maniera precisa l'azione di gioco e i giocatori in campo.

Il codice è strutturato in una serie sequenziale di Jupyter Notebook che coprono interamente il flusso di lavoro: dall'analisi esplorativa dei dati (EDA) all'elaborazione, addestramento, inferenza e valutazione finale. Il sistema utilizza modelli di intelligenza artificiale all'avanguardia come YOLOv12 per il rilevamento e la segmentazione e Segment Anything Model (SAM 2.1) per maschere ad alta precisione.

## Architettura e Flusso di Lavoro

Il progetto e strutturato in 7 notebook numerati, ideati per essere eseguiti sequenzialmente:

1. **1_EDA.ipynb (Exploratory Data Analysis):** Si occupa del caricamento delle librerie necessarie, della creazione della struttura di cartelle richiesta e del download automatico dei dataset originari tramite Kaggle API. Include, inoltre, strumenti per una prima visualizzazione grafica dei dati.
2. **2_PREPROCESSING.ipynb:** Prepara e pulisce i dati grezzi. Converte e formatta i dataset nelle specifiche richieste per l'addestramento dei modelli YOLO e per la fase di valutazione.
3. **3_YOLO.ipynb:** Gestisce la fase di rilevamento e l'addestramento sui dati preprocessati. Impiega configurazioni ottimizzate per massimizzare la precisione e velocizzare i calcoli su architetture accelerate.
4. **4_SAM.ipynb:** Implementa il Segment Anything Model per ottimizzare la segmentazione zero-shot. In questa fase il sistema perfeziona la generazione delle maschere pertinenti ai cartelloni.
5. **5_YOLO_SEG.ipynb:** Notebook specializzato sull'uso dell'architettura YOLO per la segmentazione (YOLO-Seg). Focalizzato sulla creazione di maschere per tracciare efficacemente le figure umane ed escluderle dalle modifiche ai cartelloni.
6. **6_INFERENCE.ipynb:** Costituisce il cuore dell'applicazione pratica, permettendo l'esecuzione della pipeline completa. Elabora componenti multimediali di input (frame spaziali), consolidando i rilevamenti ed applicando l'effetto "green screen" ai banner evidenziati, preservando fedelmente i giocatori in sovrapposizione.
7. **7_EVALUATION.ipynb:** Analizza oggettivamente i risultati della pipeline integrando metriche di valutazione come Intersection over Union (IoU), Precision e Recall, al fine di determinare l'efficienza della maschera complessiva e dell'esclusione target.

## Esempi Visivi e Risultati

Di seguito sono riportati i risultati visivi chiave e i relativi dati derivanti dallo sviluppo del progetto:

### Esempio Dataset YOLO
![Esempio Dataset YOLO](assets/esempio_dataset_yolo.jpg)

### Esempio Dataset SAM
![Esempio Dataset SAM](assets/esempio_dataset_sam.jpg)

### Metriche di Detection YOLO
![Metriche di Detection YOLO](assets/metriche_detection_yolo.jpg)

### Risultato Intera Pipeline
![Risultato Intera Pipeline](assets/risultato_intera_pipeline.jpg)

## Struttura del Repository

- I file nella radice principale `*.ipynb` rappresentano le fasi di esecuzione del codice (EDA, Preprocessing, YOLO, SAM, YOLO-Seg, Inferenza e Valutazione).
- Ciascuna iterazione si avvale di una cartella generata localmente nominata `project-sbs/`, progettata per memorizzare i dati transitori ed i modelli profondi generati. Tale cartella comprende:
  - `datasets/`: Posizione logica per tutti i file d'immagine grezzi ed il training-set.
  - `model_weights/`: File binari dei pesi (.pt, .pth) derivanti dall'addestramento dei modelli di deep learning.
  - `output/`: Esito delle operazioni di inferenza (immagini elaborate e output video finali).
  - `runs/`: Log temporanei, metriche e cronologia delle loss legate ad esportazioni TensorBoard e sessioni di training YOLO.

## Requisiti 

- Linguaggio Python 3.x
- Architettura hardware dotata di GPU performanti. I notebook sono stati strutturati ed ottimizzati per poter girare in ambienti cloud, considerando GPU ad altissima capacita (es. NVIDIA A100 da 80GB) per tempi ideali di computazione ad alta densita di dati.
- Autenticazione attiva tramite API Kaggle associata (inclusione di regolare file `kaggle.json`).

I framework portanti vertono su ecosistemi `Ultralytics` per YOLO e costrutti legati a PyTorch, `opencv-python-headless` per l'iterazione sulle immagini, `numpy` e `matplotlib` per le misurazioni.

## Come avviare il progetto (Setup)

1. Clonare il presente repository.
2. Assicurarsi di impostare i token di sicurezza e le credenziali di Kaggle fornendo opportunamente il file `kaggle.json`.
3. Inizializzare ed eseguire interamente l'architettura dei notebook a partire da `1_EDA.ipynb`. Tutte le dipendenze software mancanti e i percorsi interni verranno configurati automaticamente al primo ciclo di esecuzione.
