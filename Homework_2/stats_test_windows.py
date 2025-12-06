
import pygame
import time
from multiprocessing import Queue

class StatsTestWindow:
    def __init__(self, q: Queue, env_name="Environment"):
        self.q = q
        self.env_name = env_name

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((450, 450))
        pygame.display.set_caption(f"Stats Test → {self.env_name}")
        font = pygame.font.SysFont("Arial", 22)
        clock = pygame.time.Clock()
        running = True

        # valori iniziali
        episode = 0
        reward = 0
        mean_reward = 0
        best_reward = 0
        steps = 0
        std_reward = 0
        completed = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Leggi nuovi dati dalla coda
            while not self.q.empty():
                data = self.q.get()

                episode = data.get("episode", episode)
                reward = data.get("reward", reward)
                steps = data.get("steps", steps)
                mean_reward = data.get("mean_reward", mean_reward)
                std_reward = data.get("std_reward", std_reward)
                best_reward = data.get("best_reward", best_reward)
                completed = data.get("completed", completed)

            # Render
            screen.fill((25, 25, 25))

            lines = [
                f"Environment: {self.env_name}",
                f"Episode: {episode}",
                f"Reward Episode: {reward:.2f}",
                f"Steps Episode: {steps}",
                f"Best Reward: {best_reward:.2f}",
                f"Mean Reward: {mean_reward:.2f}",
                f"Std Dev: {std_reward:.2f}",
                f"Episodes Completed: {completed}",
            ]

            y = 40
            for line in lines:
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (30, y))
                y += 40

            pygame.display.flip()
            clock.tick(30)
            time.sleep(0.05)

        pygame.quit()
