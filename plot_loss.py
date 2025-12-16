import json
import matplotlib.pyplot as plt
import sys
import os
import argparse

def plot_loss(log_file):
    if not os.path.exists(log_file):
        print(f"Hata: Dosya bulunamadı -> {log_file}")
        return

    steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    print(f"Veriler okunuyor: {log_file}")

    with open(log_file, 'r') as f:
        # Dosya uzantısına göre okuma stratejisi
        if log_file.endswith('.json'):
            # Eğer trainer_state.json ise (HuggingFace standardı)
            data = json.load(f)
            if "log_history" in data:
                history = data["log_history"]
            else:
                history = [data] # Tek parça json ise
            
            for entry in history:
                if "loss" in entry:
                    steps.append(entry["step"])
                    train_losses.append(entry["loss"])
                if "eval_loss" in entry:
                    eval_steps.append(entry["step"])
                    eval_losses.append(entry["eval_loss"])
        else:
            # Eğer .jsonl ise (Satır satır log - Bizim scriptin çıktısı)
            for line in f:
                try:
                    data = json.loads(line)
                    if "loss" in data and data["loss"] is not None:
                        steps.append(data["step"])
                        train_losses.append(data["loss"])
                    if "eval_loss" in data and data["eval_loss"] is not None:
                        eval_steps.append(data["step"])
                        eval_losses.append(data["eval_loss"])
                except json.JSONDecodeError:
                    continue

    if not steps:
        print("Hata: Dosyada çizilecek 'loss' verisi bulunamadı.")
        return

    # Grafik Ayarları
    plt.figure(figsize=(12, 7))

    # Train Loss
    plt.plot(steps, train_losses, 
             label='Training Loss', 
             color='#1f77b4', 
             linestyle='-', 
             marker='.',       # Her veri noktasında işaret (Sıklık veriye bağlı)
             alpha=0.8)
    
    # Validation Loss
    if eval_steps:
        plt.plot(eval_steps, eval_losses, 
                 label='Validation Loss', 
                 color='#d62728', 
                 linestyle='--', 
                 marker='o',       # Belirgin yuvarlak
                 linewidth=2)

    plt.title('Training Loss Analysis', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Dosya ismine göre kaydet
    output_img = log_file.rsplit('.', 1)[0] + '_graph.png'
    plt.savefig(output_img, dpi=300)
    print(f"✅ Grafik kaydedildi: {output_img}")
    plt.show()

if __name__ == "__main__":
    # Kullanım: python plot_loss.py <dosya_yolu>
    # Eğer argüman verilmezse varsayılan olarak 'training_log.jsonl' arar
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = "training_log.jsonl"
        
    plot_loss(target_file)