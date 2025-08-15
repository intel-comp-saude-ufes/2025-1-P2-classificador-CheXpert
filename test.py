import os
import torch

class CheckpointManager:
    def __init__(self, save_dir: str, save_name: str = "model.pt", mode: str = "min", monitor: str = "val_loss"):
        """
        Gerencia salvamento e carregamento de checkpoints.

        Args:
            save_dir (str): Diretório onde salvar os arquivos.
            save_name (str): Nome base do arquivo de checkpoint.
            mode (str): "min" para monitorar menor valor (ex: loss), "max" para maior (ex: acc).
            monitor (str): Nome da métrica monitorada.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.save_name = save_name
        self.mode = mode
        self.monitor = monitor
        self.best_value = float("inf") if mode == "min" else -float("inf")

        self.last_path = os.path.join(save_dir, f"last_{save_name}")
        self.best_path = os.path.join(save_dir, f"best_{save_name}")

    def is_better(self, current):
        if self.mode == "min":
            return current < self.best_value
        else:
            return current > self.best_value

    def save(self, model, optimizer, epoch, history, current_value):
        """
        Salva o último checkpoint e, se for o melhor, salva também o best.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history.get_inner_dict(),
            "monitor_value": current_value,
            "best_value": self.best_value
        }

        # Sempre salva o último
        torch.save(checkpoint, self.last_path)

        # Se for melhor, salva como best
        if self.is_better(current_value):
            self.best_value = current_value
            torch.save(checkpoint, self.best_path)
            return True  # indica que é o melhor até agora
        
        return False

    def load(self, model, optimizer=None, best=False, device="cpu"):
        """
        Carrega um checkpoint no modelo (e otimizador se passado).
        """
        path = self.best_path if best else self.last_path
        if not os.path.exists(path):
            print(f"[CheckpointManager] Nenhum checkpoint encontrado em {path}")
            return 0, None, None  # epoch, history, monitor_value

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_value = checkpoint.get("best_value", self.best_value)
        
        return (
            checkpoint["epoch"] + 1,  # próxima época
            checkpoint.get("history", None),
            checkpoint.get("monitor_value", None)
        )
