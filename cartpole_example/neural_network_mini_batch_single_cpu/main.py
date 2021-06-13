import policy_gradient_class


if __name__ == '__main__':

    env = 'CartPole'
    use_cuda = True
    
    policy_gradient = policy_gradient_class.PolicyGradient(problem=env, use_cuda=use_cuda)
    policy_gradient.solve_environment()