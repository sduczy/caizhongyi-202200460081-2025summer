pragma circom 2.1.4;

include "/home/seed/Desktop/circom/parser/circomlib-master/circuits/poseidon.circom";

template PoseidonHash2() {
    signal input in[2];
    signal output out;

    component h = Poseidon(2);

    for (var i = 0; i < 2; i++) {
        h.inputs[i] <== in[i];
    }

    out <== h.out;
}

component main = PoseidonHash2();
