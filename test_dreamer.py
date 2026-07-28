import pathlib
import ruamel.yaml as yaml
import numpy as np
import elements

# Importiamo esplicitamente il file dell'agente
from dreamerv3 import agent as agent_lib 

def main():
    print("🚀 Inizializzazione Smoke Test per DreamerV3...")

    # 1. Trova e carica le configurazioni dal file configs.yaml di DreamerV3
    config_path = pathlib.Path(agent_lib.__file__).parent / 'configs.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato in: {config_path}")
        
    with open(config_path, 'r') as f:
        configs = yaml.YAML(typ='safe').load(f)

    # Inizializziamo il config con i default letti dal file
    config = elements.Config(configs['defaults'])
    
    # Applichiamo un preset se esiste
    if 'size12m' in configs:
        config = config.update(configs['size12m'])

    try:
        config = config.update({
            'logdir': './logdir/smoke_test',
            'run.steps': 100,
            'run.log_every': 20,
        })
    except KeyError:
        print("⚠️ Chiavi del runner non supportate dal config dell'Agent (ignorate).")

    # Il default 'cuda' fallisce su macchine senza GPU NVIDIA (es. Mac): forziamo la CPU.
    config = config.update({'jax.platform': 'cpu'})

    print(f"✅ Configurazione caricata correttamente da {config_path.name}!")
    
    # Controllo per evitare errori se 'batch_size' e 'jax' mancano nel config base
    if 'batch_size' in config:
        print(f"   - Batch Size: {config.batch_size}")
    if 'jax' in config and 'precision' in config.jax:
        print(f"   - Precisione JAX: {config.jax.precision}")

    # 2. Test rapido creazione Agente (spazio di osservazione dummy)
    # Per convenzione embodied, obs_space deve includere is_first/is_last/
    # is_terminal/reward, e act_space deve includere 'reset'.
    obs_space = {
        'image': elements.Space(np.uint8, shape=(64, 64, 3)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    # 'reset' fa parte dell'act_space dell'ambiente ma va escluso prima di
    # passarlo all'Agent, esattamente come fa dreamerv3/main.py:make_agent().
    act_space = {'action': elements.Space(np.float32, shape=(4,))}

    # Agent si aspetta un config "piatto" con le chiavi di config.agent portate
    # al livello superiore, esattamente come fa dreamerv3/main.py:make_agent().
    agent_config = elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    )

    # Passiamo gli argomenti alla classe Agent trovata nel modulo 'agent_lib'
    agent = agent_lib.Agent(obs_space, act_space, agent_config)
    print("✅ Agente e World Model JAX creati con successo!")

if __name__ == '__main__':
    main()