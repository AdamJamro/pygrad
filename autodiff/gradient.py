from numpy import ones, zeros_like

from autodiff.variable import Variable, Tape, get_tape_stack


def grad(
    loss_variable: Variable, tape_records: list[Tape] = get_tape_stack(), desired_results: list[Variable] | None = None
) -> dict[Variable, Variable]:
    """
    Computes gradients of the loss with respect to each Variable in the computation,
    graph stores the variables in a dLoss_d lookup map which effectively works out to make dLoss_d[x] equal to dLoss/dx

    loss_variable:
        The top node of the computation graph representing the loss.
        If the loss is not a scalar, it's implicitly set to be an array of ones with the same shape as the loss.

    desired_results:
        # TODO currently serves no optimization
        Selects what we exclude from the gradient result list.
        If desired_results is None, gradients for all variables in the computation graph will be computed.

    If the tape does not contain a variable,
    we consider its gradient None.
    """

    dLoss_d : dict[Variable, Variable] = {loss_variable: Variable(ones(()))}

    def prune_unused_outputs(outputs: tuple[Variable, ...]) -> tuple[Variable | None, ...]:
        return tuple((dLoss_d[output] if output in dLoss_d else None) for output in outputs)

    for tape_record in reversed(tape_records):
        dLoss_dOutputs = prune_unused_outputs(tape_record.outputs)

        if all(dL_dOutput is None for dL_dOutput in dLoss_dOutputs):
            continue # prune paths equal to zero vectors

        # perform chain rule propagation specific to each compute
        dLoss_dInputs = tape_record.back_fn(dLoss_dOutputs)

        # the dag computation can have different shapes than simple MLP's so we actually sum all the gradients
        for tape_input, dL_dInput in zip(tape_record.inputs, dLoss_dInputs):
            # we could have used defaultdict(lambda x:0) but this way we keep the notion of what was actually used
            if tape_input not in dLoss_d:
                dLoss_d[tape_input] = dL_dInput
            else:
                dLoss_d[tape_input] += dL_dInput

    # print some information to understand the values of each intermediate
    for name, value in dLoss_d.items():
        print(f'd{loss_variable.name}_d{name} = {value.name}')

    return dLoss_d
