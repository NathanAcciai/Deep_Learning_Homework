# stats_window.py
import pygame
import time
from multiprocessing import Queue

def run_stats_window(q: Queue):
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Statistiche REINFORCE")
    font = pygame.font.SysFont("Arial", 22)
    clock = pygame.time.Clock()
    running = True

    # valori iniziali
    episode = 0
    reward = 0
    best = 0
    success=0
    failure=0
    episode_val=0
    rewards_val=0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # leggi dalla coda se ci sono nuovi dati
        while not q.empty():
            data = q.get()
            episode = data.get("episode", episode)
            reward = data.get("reward", reward)
            best = data.get("best", best)
            success= data.get("success", success)
            failure=data.get("failure", failure)
            episode_val= data.get("episode_val",episode_val)
            rewards_val=data.get("reward_val",rewards_val)
        # disegna
        screen.fill((30, 30, 30))
        text1 = font.render(f"Episode: {episode}", True, (255, 255, 255))
        text2 = font.render(f"Reward: {reward:.2f}", True, (255, 255, 255))
        text3 = font.render(f"Best Reward: {best:.2f}", True, (255, 255, 255))
        text4 = font.render(f"Success: {success:.2f}", True, (255, 255, 255))
        text5 = font.render(f"Failure: {failure:.2f}", True, (255, 255, 255))
        text6 = font.render(f"Episode Validation: {episode_val:.2f}", True, (255, 255, 255))
        text7 = font.render(f"Reward Validation: {rewards_val:.2f}", True, (255, 255, 255))
        
        screen.blit(text1, (30, 50))
        screen.blit(text2, (30, 90))
        screen.blit(text3, (30, 130))
        screen.blit(text4, (30, 170))
        screen.blit(text5, (30, 210))
        screen.blit(text6, (30, 250))
        screen.blit(text7, (30, 290))

        pygame.display.flip()
        clock.tick(30)
        time.sleep(0.1)

    pygame.quit()
