from stable_baselines3 import PPO
from environments.walk_environment import WalkEnvironmentV0
from environments.jump_environment import JumpEnvironmentV0
from argparse import ArgumentParser

def test_model(env, motion):
    # Cargar el modelo correspondiente al tipo de movimiento
    model_path = f"/home/mau/Documentos/Escuela/TT/app/models/{motion}/best_model/best_model"
    model = PPO.load(model_path)
    print (model.policy)
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=False)

        print(f"Action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print("obs", obs)
        print("reward", reward)
        print("terminated", terminated)
        print("truncated", truncated)
        print("info", info)

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('motion', choices=["walk", "jump"], help="Specify the motion type: 'walk' or 'jump'")
    args = parser.parse_args()

    # Elegir el entorno según el tipo de movimiento
    if args.motion == "walk":
        env = WalkEnvironmentV0(render_mode="human")
    elif args.motion == "jump":
        env = JumpEnvironmentV0(render_mode="human")

    # Llamar a la función de prueba con el entorno y tipo de movimiento especificado
    test_model(env, args.motion)
