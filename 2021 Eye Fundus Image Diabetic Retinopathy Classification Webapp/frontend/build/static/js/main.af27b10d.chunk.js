(window.webpackJsonp=window.webpackJsonp||[]).push([[0],{20:function(e,a,t){e.exports=t(52)},25:function(e,a,t){},27:function(e,a,t){e.exports=t.p+"static/media/logo.5d5d9eef.svg"},28:function(e,a,t){},52:function(e,a,t){"use strict";t.r(a);var n=t(0),i=t.n(n),r=t(12),l=t.n(r),o=(t(25),t(13)),c=t(14),s=t(18),d=t(15),m=t(19),h=t(2),u=(t(27),t(28),t(16)),g=t.n(u),p=t(17),v=t.n(p),w=function(e){function a(){var e;return Object(o.a)(this,a),(e=Object(s.a)(this,Object(d.a)(a).call(this))).state={previewImageUrl:!1,imageHeight:400,chart_options:{chart:{id:"basic-bar"},xaxis:{categories:["Normal","Leve","Moderada","Severa"]}},chart_series:[{name:"prob",data:[0,0,0,0]}]},e.generatePreviewImageUrl=e.generatePreviewImageUrl.bind(Object(h.a)(Object(h.a)(e))),e.handleChange=e.handleChange.bind(Object(h.a)(Object(h.a)(e))),e.uploadHandler=e.uploadHandler.bind(Object(h.a)(Object(h.a)(e))),e}return Object(m.a)(a,e),Object(c.a)(a,[{key:"generatePreviewImageUrl",value:function(e,a){var t=new FileReader;t.readAsDataURL(e);t.onloadend=function(e){return a(t.result)}}},{key:"handleChange",value:function(e){var a=this,t=e.target.files[0];t&&(this.setState({imageFile:t}),this.generatePreviewImageUrl(t,function(e){a.setState({previewImageUrl:e,imagePrediction:""})}))}},{key:"uploadHandler",value:function(e){var a=this,t=new FormData;t.append("file",this.state.imageFile);var n=performance.now();g.a.post("http://medical.neural-solutions.com:5000/upload",t).then(function(e,t){t=e.data,a.setState({imagePrediction:t.label,previewImageUrl:"data:image/jpeg;base64,"+t.img,chart_series:[{data:[t.c0,t.c1,t.c2,t.c3]}]});var i=performance.now();console.log("The time it took to predict the image "+(i-n)+" milliseconds.")})}},{key:"render",value:function(){return i.a.createElement("div",{className:"App"},i.a.createElement("header",{className:"App-header"},i.a.createElement("h3",null,"Sistema de Ayuda al Diagn\xf3stico. ",i.a.createElement("br",null),"Evaluaci\xf3n del grado de Retinopat\xeda Diab\xe9tica"),i.a.createElement("div",{className:"App-upload"},i.a.createElement("div",{id:"cotainer"},i.a.createElement("p",null,"Imagen de fondo de ojo a evaluar:"),i.a.createElement("div",{id:"left"},i.a.createElement("div",null,this.state.previewImageUrl&&i.a.createElement("img",{height:this.state.imageHeight,alt:"",src:this.state.previewImageUrl})),i.a.createElement("div",null,i.a.createElement("input",{type:"file",name:"file",onChange:this.handleChange})),i.a.createElement("div",null,i.a.createElement("input",{type:"submit",onClick:this.uploadHandler}))),i.a.createElement("div",{id:"center"}),i.a.createElement("div",{id:"right"},i.a.createElement("div",null,this.state.imagePrediction&&i.a.createElement("p",null,this.state.imagePrediction)),i.a.createElement("div",null,this.state.imagePrediction&&i.a.createElement(v.a,{options:this.state.chart_options,series:this.state.chart_series,type:"bar",width:this.state.imageHeight})))))))}}]),a}(n.Component);Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));l.a.render(i.a.createElement(w,null),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then(function(e){e.unregister()})}},[[20,2,1]]]);
//# sourceMappingURL=main.af27b10d.chunk.js.map