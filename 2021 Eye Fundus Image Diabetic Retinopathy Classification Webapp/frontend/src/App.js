import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import axios from 'axios';
import Chart from 'react-apexcharts'

class App extends Component {

  // Constructor
  constructor() {
    super()
    this.state = {
      previewImageUrl: false,
      imageHeight: 400,
      chart_options: {
        chart: {
          id: 'basic-bar',
        },
        xaxis: {
          categories: ['Normal', 'Leve', 'Moderada', 'Severa'
          ],
        }
      },
      chart_series: [{
        name: "prob",
        data: [0, 0, 0, 0]
      }],
    }
    this.generatePreviewImageUrl = this.generatePreviewImageUrl.bind(this)
    this.handleChange = this.handleChange.bind(this)
    this.uploadHandler = this.uploadHandler.bind(this)
  }

    // Function for previewing the chosen image
    generatePreviewImageUrl(file, callback) {
      const reader = new FileReader()
      const url = reader.readAsDataURL(file)
      reader.onloadend = e => callback(reader.result)
    }

    // Event handler when image is chosen
    handleChange(event) {
      const file = event.target.files[0]
      
      // If the image upload is cancelled
      if (!file) {
        return
      }

      this.setState({imageFile: file})
      this.generatePreviewImageUrl(file, previewImageUrl => {
            this.setState({
              previewImageUrl,
              imagePrediction:""
            })
          })
    }

    // Function for sending image to the backend
    uploadHandler(e) {
    var self = this;
    const formData = new FormData()
    formData.append('file', this.state.imageFile)
    
    var t0 = performance.now();
    axios.post('http://medical.neural-solutions.com:5000/upload', formData)
    .then(function(response, data) {
            data = response.data;
            self.setState(
              {
                imagePrediction: data["label"], 
                previewImageUrl: "data:image/jpeg;base64," + data["img"], 
                chart_series: [{
                  data: [data['c0'], data['c1'], data['c2'], data['c3']]
                }]
              })
            var t1 = performance.now();
            console.log("The time it took to predict the image " + (t1 - t0) + " milliseconds.")
        })
    }

  render() {
    return (
      <div className="App">
        <header className="App-header">
        <h3>
            Sistema de Ayuda al Diagnóstico. <br />
            Evaluación del grado de Retinopatía Diabética
	      </h3>          
        <div className="App-upload">
          <div id="cotainer">
            <p>
              Imagen de fondo de ojo a evaluar:
            </p>             
            <div id="left">             
              {/* Field for previewing the chosen image */}
              <div>
                { this.state.previewImageUrl &&
                <img height={this.state.imageHeight} alt="" src={this.state.previewImageUrl} />
                }
              </div>  
              {/* Button for choosing an image */}
              <div>
                <input type="file" name="file" onChange={this.handleChange} />
              </div>

              {/* Button for sending image to backend */}
              <div>
                <input type="submit" onClick={this.uploadHandler} />
              </div>                          
            </div>
            <div id="center"></div>
            <div id="right">
              {/* Text for model prediction */}
              <div>
                { this.state.imagePrediction &&
                  <p>{this.state.imagePrediction}</p>
                }
              </div>

              {/* Probability Graph */}
              <div>
                { this.state.imagePrediction &&              
                    <Chart
                      options={this.state.chart_options}
                      series={this.state.chart_series}
                      type="bar"
                      width={this.state.imageHeight}
                    />
                }
              </div>
            </div>
          </div>
        </div>
        </header>
      </div>
    );
  }
}

export default App;
