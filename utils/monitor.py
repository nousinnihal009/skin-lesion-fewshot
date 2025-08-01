from tqdm import tqdm


class ProgressMonitor:
    def __init__(self, total_episodes, desc="Training"):
        self.pbar = tqdm(total=total_episodes, desc=desc)

    def update(self, episode, loss, acc):
        self.pbar.set_postfix({
            "Loss": f"{loss:.4f}",
            "Acc": f"{acc*100:.2f}%"
        })
        self.pbar.update(1)

    def close(self):
        self.pbar.close()
