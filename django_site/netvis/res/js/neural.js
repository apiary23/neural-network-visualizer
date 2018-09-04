'use strict';

const container = document.getElementById('myNet'),
      networkData = {},
      eNodes = new vis.DataSet([]),
      eEdges = new vis.DataSet([]),
      eData = {eNodes, eEdges},
      eOptions = {groups: {}},
      network = new vis.Network(container, eData, eOptions);

const randHex = () => '0123456789abcdef'[Math.floor(Math.random()*16)],
      randColor = () => `#${randHex()}${randHex()}${randHex()}`;

const train = function(batchSize) {
    let nodes = new vis.DataSet([]),
        edges = new vis.DataSet([]),
        data = {nodes, edges},
        options = {groups: {}, physics: {enabled: false}};

    fetch(`./trainNet?batchSize=${batchSize}`)
    .then(result => {
        return result.json(); 
    }).then(result => {

        let nCounter = 0;
        result.topology.forEach((count, index) => {
            [...Array(count).keys()].forEach((_, subind) => {
                data.nodes.add({id: nCounter,
                                group: `g${index}`,
                                x: index*150,
                                y: subind*100 - count*50,
                                fixed: true});            
                nCounter += 1;
            });
        });
        Object.keys(result.arcs).forEach(fromInd => {
            Object.keys(result.arcs[fromInd]).forEach(toInd => {
                data.edges.add({from: fromInd,
                                to: toInd,
                                width: result.arcs[fromInd][toInd]});
            });
        });
        network.setData(data);
        network.setOptions(options);
        
    });

};

fetch("./getNet")
    .then(result => {
        console.log(`status: ${result.status}`);
        train(1);
    });



document.getElementById('trainBtn').addEventListener('click', () => {
    train(100);
});


