import policy_gradient_class


if __name__ == '__main__':

    use_cuda = True
    
    policy_gradient = policy_gradient_class.PolicyGradient(use_cuda=use_cuda)
    policy_gradient.solve_environment()