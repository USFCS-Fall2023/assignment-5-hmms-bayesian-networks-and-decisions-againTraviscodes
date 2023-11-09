from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination


def belief_networks_carnet():
    car_model = BayesianNetwork(
        [
            ("Battery", "Radio"),
            ("Battery", "Ignition"),
            ("Ignition", "Starts"),
            ("Gas", "Starts"),
            ("KeyPresent", "Starts"),
            ("Starts", "Moves")
        ]
    )

    # Defining the parameters using CPT
    from pgmpy.factors.discrete import TabularCPD

    cpd_battery = TabularCPD(
        variable="Battery",
        variable_card=2,
        values=[[0.70], [0.30]],
        state_names={"Battery": ['Works', "Doesn't work"]},
    )
    cpd_gas = TabularCPD(
        variable="Gas",
        variable_card=2,
        values=[[0.40], [0.60]],
        state_names={"Gas": ['Full', "Empty"]},
    )
    cpd_radio = TabularCPD(
        variable="Radio",
        variable_card=2,
        values=[[0.75, 0.01], [0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Radio": ["turns on", "Doesn't turn on"],
                     "Battery": ['Works', "Doesn't work"]}
    )
    cpd_ignition = TabularCPD(
        variable="Ignition",
        variable_card=2,
        values=[[0.75, 0.01], [0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Ignition": ["Works", "Doesn't work"],
                     "Battery": ['Works', "Doesn't work"]}
    )
    cpd_starts = TabularCPD(
        variable="Starts",
        variable_card=2,
        values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
        evidence=["Ignition", "Gas", "KeyPresent"],
        evidence_card=[2, 2, 2],
        state_names={"Starts": ['yes', 'no'],
                     "Ignition": ["Works", "Doesn't work"],
                     "Gas": ['Full', "Empty"],
                     "KeyPresent": ["yes", "no"]},
    )
    cpd_moves = TabularCPD(
        variable="Moves",
        variable_card=2,
        values=[[0.8, 0.01], [0.2, 0.99]],
        evidence=["Starts"],
        evidence_card=[2],
        state_names={"Moves": ["yes", "no"],
                     "Starts": ['yes', 'no']}
    )
    cpd_keypresent = TabularCPD(
        variable="KeyPresent",
        variable_card=2,
        values=[[0.7], [0.3]],
        state_names={"KeyPresent": ["yes", "no"]}
    )

    # Associating the parameters with the model structure
    car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keypresent)
    car_infer = VariableElimination(car_model)
    # print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

    # Part 2
    n02_1 = car_infer.query(variables=['Battery'], evidence={'Moves': 'no'})
    n02_2 = car_infer.query(variables=['Starts'], evidence={'Radio': "Doesn't turn on"})
    n02_3_1 = car_infer.query(variables=['Radio'], evidence={'Battery': 'Works'})  # , 'Gas': 'Empty'
    n02_3_2 = car_infer.query(variables=['Radio'], evidence={'Battery': 'Works', 'Gas': 'Full'})
    n02_4_1 = car_infer.query(variables=['Moves'], evidence={'Ignition': 'Works'})  # , 'Gas': 'Empty'
    n02_4_2 = car_infer.query(variables=['Moves'], evidence={'Ignition': 'Works', 'Gas': 'Full'})
    n02_5 = car_infer.query(variables=['Starts'], evidence={'Radio': 'turns on', 'Gas': 'Full'})

    print("Part 2")
    print("Moves given Battery Doesn't work")
    print(n02_1)
    print("Starts given Radio Doesn't turn on")
    print(n02_2)
    print("Radio given Battery Works")  # , Gas Empty
    print(n02_3_1)
    print("Radio given Battery Works, Gas Full")
    print(n02_3_2)
    values_changed = get_changes(n02_3_1.values, n02_3_2.values)
    print(f"The values change given Gas Full: {values_changed}")
    print("Moves given Ignition Works")  # , Gas Empty
    print(n02_4_1)
    print("Moves given Ignition Works, Gas Full")
    print(n02_4_2)
    values_changed = get_changes(n02_4_1.values, n02_4_2.values)
    print(f"The values change given Gas Full: {values_changed}")
    print("Starts given Radio turns on, Gas Full")
    print(n02_5)

    # Personal testing for part 3, question 3
    n03_1 = car_infer.query(variables=["Moves"], evidence={"Ignition": "Works", "KeyPresent": "yes"})
    n03_2 = car_infer.query(variables=["Moves"], evidence={"Ignition": "Works", "KeyPresent": "yes", "Gas": "Full"})

    print("Part 3")
    print(n03_1)
    print("Given Gas Full")
    print(n03_2)


def get_changes(l1, l2):
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return True
    return False


if __name__ == '__main__':
    belief_networks_carnet()
