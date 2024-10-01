const tf = require('@tensorflow/tfjs-node')
const fs = require('fs')
const path = require('path')
const {glob} = require('glob')


function fileToTensor(filePath,size){
  const rawimage = fs.readFileSync(filePath)
  const imageTensor = tf.node.decodeImage(rawimage,3)
  const resizedTensor = tf.image.resizeBilinear(imageTensor,size)
  const normalizedTensor = tf.cast(resizedTensor.div(tf.scalar(255)), dtype = 'float32');
  return normalizedTensor
}

function trainFolderToTensors(dirPath,size) {
  return new Promise((resolve, reject) => {
    const XS = []
    const YS = []
    const dirs = []
    console.log('Identifying Image List')
    glob(`${dirPath}/*/*.@(png|jpeg|jpg|bmp)`)
    .then(files => {
      console.log(`${files.length} Files Found`)
      console.log('Now converting to tensors')
      files.forEach((file) => {
        // console.log(file)
        const dir = path.basename(path.dirname(file))
        if (!dirs.includes(dir)) {
          dirs.push(dir)
        }
        const answer = dirs.indexOf(dir)
        const imageTensor = fileToTensor(file,size)
        // console.log(imageTensor.shape)
        YS.push(answer)
        XS.push(imageTensor)
      })
      // Shuffle the data (keep XS[n] === YS[n])
      function shuffleCombo(array, array2) {
        let counter = array.length
        console.assert(array.length === array2.length)
        let temp, temp2
        let index = 0
        while (counter > 0) {
          index = (Math.random() * counter) | 0
          counter--
          temp = array[counter]
          temp2 = array2[counter]
          array[counter] = array[index]
          array2[counter] = array2[index]
          array[index] = temp
          array2[index] = temp2
        }
      }
      shuffleCombo(XS, YS)
      
      console.log('Stacking')
      const X = tf.stack(XS)
      const Y = tf.oneHot(YS, dirs.length)

      console.log('Images all converted to tensors:')
      console.log('X', X.shape)
      console.log('Y', Y.shape)

      // cleanup
      tf.dispose(XS)

      resolve([X, Y, dirs])
    })
    .catch(error => {
      console.error('Failed to access files', error)
      reject()
      process.exit(1)
    })
  })
}

function verifyFolderToTensors(dirPath,size) {
  return new Promise((resolve, reject) => {
    const XS = []
    const YS = []
    console.log('Identifying Image List')
    glob(`${dirPath}/*.@(png|jpeg|jpg|bmp)`)
    .then(files => {
      console.log(`${files.length} Files Found`)
      console.log('Now converting to tensors')
      files.forEach((file) => {
        // console.log(file)
        const name = path.basename(file)
        const imageTensor = fileToTensor(file,size)
        console.log(name,imageTensor.shape)
        YS.push(name)
        XS.push(imageTensor)
      })
      
      console.log('Stacking')
      const X = tf.stack(XS)

      console.log('Images all converted to tensors:')
      console.log('X', X.shape)
      console.log('Y', YS)

      // cleanup
      tf.dispose(XS)

      resolve([X, YS])
    })
    .catch(error => {
      console.error('Failed to access files', error)
      reject()
      process.exit(1)
    })
  })
}

async function learnTransferModel(folderPath,socket=null) {
  console.log('Loading images - this may take a while...')
  if(socket){socket.emit('log','loading images')}
  const [X,Y,dirs] = await trainFolderToTensors(folderPath,[224,224])
  console.log(dirs)

  console.log('Loading model')
  if(socket){socket.emit('log','loading model')}
  // Load feature model
  // const featureModel = await tf.loadGraphModel('https://www.kaggle.com/models/google/mobilenet-v2/TfJs/140-224-feature-vector/3', {fromTFHub: true});
  const featureModelPath = 'c:/users/kosan/desktop/sotsuken/keras_mobilenet-v2_featurevec/model.json'
  const featureModel = await tf.loadLayersModel('file://'+featureModelPath);
  // featureModel:tf.Model

  const transferModel = tf.sequential({
    // Create NN
    layers: [
      tf.layers.dense({
        inputShape: [1792],
        units: 64,
        activation: 'relu',
      }),
      tf.layers.dense({ units: dirs.length, activation: 'softmax' }),
    ],
  })

  console.log('Creating features from images - this may take a while...')
  if(socket){socket.emit('log','creating features')}

  const featureX = featureModel.predict(X)
  // Push data through feature detection
  console.log(`Features stack ${featureX.shape}`)
  if(socket){socket.emit('log',`Features stack ${featureX.shape}`)}

  transferModel.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })

  console.log(transferModel.evaluate(featureX,Y)[1].dataSync());

  const history = await transferModel.fit(featureX, Y, {
    epochs: 100,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (false) {
          model.stopTraining = true
        }
        console.log(`epoch:${epoch} acc:${logs.acc}`)
        if(socket){socket.emit('updateProgress',epoch)}
      }
    },
    verbose:false
  })
  // console.log(history)
  console.log('learned!')
  if(socket){socket.emit('log','learned')}

  console.log('saving model')
  if(socket){socket.emit('log','saving model')}
  modelPath = path.dirname(folderPath)
  modelPath = path.join(modelPath,'model')
  console.log(modelPath)
  await transferModel.save('file://'+modelPath)
  classPath = path.join(modelPath,'classes.json')
  fs.writeFileSync(classPath,JSON.stringify(dirs))
  console.log('model saved')
  if(socket){socket.emit('log','model saved')}
  return [transferModel,dirs]
}

async function validateImages(folderPath) {
  const hrstart = process.hrtime()

  console.log('Loading images - this may take a while...')
  const [X, names] = await verifyFolderToTensors(folderPath,[224,224])
  
  console.log('Loading model')
  // Load feature model
  // const featureModel = await tf.loadGraphModel('https://www.kaggle.com/models/google/mobilenet-v2/TfJs/140-224-feature-vector/3', {fromTFHub: true});
  const featureModelPath = 'c:/users/kosan/desktop/sotsuken/keras_mobilenet-v2_featurevec/model.json'
  const featureModel = await tf.loadLayersModel('file://'+featureModelPath);
  // featureModel:tf.Model
  modelPath = path.dirname(folderPath)
  modelPath = path.dirname(modelPath)
  modelPath = path.join(modelPath,'model')
  classPath = path.join(modelPath,'classes.json')
  modelPath = path.join(modelPath,'model.json')
  const transferModel = await tf.loadLayersModel('file://'+modelPath)
  const classes = JSON.parse(fs.readFileSync(classPath).toString())
  // console.log(classes)

  console.log('Creating features from images - this may take a while...')

  const featureX = featureModel.predict(X)
  // Push data through feature detection
  console.log(`Features stack ${featureX.shape}`)

  console.log('Predicting...')
  const predicted = transferModel.predict(featureX)
  // Push data through feature detection
  console.log(`Features stack ${predicted.shape}`)
  confidences = predicted.arraySync()
  const images = []
  for (const [i, name] of Object.entries(names)){
    images.push({name:name,confidence:confidences[i]})
  }
  console.log(images)
  const hrend = process.hrtime(hrstart)
  const execTime_ms = hrend[0]*1e3 + hrend[1]*1e-6
  const result = {
    classes:classes,
    images:images,
    execTime_ms:execTime_ms
  }
  return result
}

module.exports = {
  learnTransferModel:learnTransferModel,
  validateImages:validateImages,
}


const run = async() => {
  await learnTransferModel('c:/users/kosan/desktop/sotsuken/system/training-data');
  await validateImages('c:/users/kosan/desktop/sotsuken/system/verify-data/verify1');
};
// run()
console.log('we can reach here!!')