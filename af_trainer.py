import trainer as base_trainer


def train_disc_af_ddqn_ray(
    problem_name="ZDT1",
    dim=30,
    epoch=None,
    gamma=None,
    reward_scheme=1,
    surrogate_model="kan",
    training_set=1,
    num_workers=None,
    surrogate_nsga_steps=100,
    updates_per_epoch=None,
    device=None,
    rollout_device="cpu",
    surrogate_device="cpu",
    use_ray=False,
):
    return base_trainer.train_disc_ddqn_ray(
        problem_name=problem_name,
        dim=dim,
        epoch=epoch,
        gamma=gamma,
        reward_scheme=reward_scheme,
        surrogate_model=surrogate_model,
        training_set=training_set,
        num_workers=num_workers,
        surrogate_nsga_steps=surrogate_nsga_steps,
        updates_per_epoch=updates_per_epoch,
        device=device,
        rollout_device=rollout_device,
        surrogate_device=surrogate_device,
        use_ray=use_ray,
        agent_name="disc_af",
    )


if __name__ == "__main__":
    args = base_trainer.parse_args()
    train_disc_af_ddqn_ray(
        problem_name=args.problem,
        dim=int(args.dim),
        epoch=args.epoch,
        gamma=args.gamma,
        reward_scheme=int(args.reward_scheme),
        surrogate_model=str(args.surrogate_model),
        training_set=int(args.training_set),
        num_workers=args.num_workers,
        surrogate_nsga_steps=int(args.surrogate_nsga_steps),
        updates_per_epoch=args.updates_per_epoch,
        device=args.device,
        rollout_device=str(args.rollout_device),
        surrogate_device=str(args.surrogate_device),
        use_ray=bool(args.ray),
    )
