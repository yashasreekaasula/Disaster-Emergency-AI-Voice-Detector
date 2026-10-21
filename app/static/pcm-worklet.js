/* Streaming FIR low-pass and fractional interpolation to mono 16 kHz PCM16. */
class PCMProcessor extends AudioWorkletProcessor {
  constructor(){super();this.step=sampleRate/16000;this.index=0;this.next=0;this.prev=0;this.ring=new Float32Array(63);this.cursor=0;this.filter=new Float32Array(63);let sum=0;const cutoff=Math.min(.45,7200/sampleRate);
    for(let i=0;i<63;i++){const x=i-31,v=(x===0?2*cutoff:Math.sin(2*Math.PI*cutoff*x)/(Math.PI*x))*(.5-.5*Math.cos(2*Math.PI*i/62));this.filter[i]=v;sum+=v;}for(let i=0;i<63;i++)this.filter[i]/=sum;this.pcm=new Int16Array(1600);this.used=0;this.energy=0;}
  process(inputs){const channels=inputs[0];if(!channels||!channels.length)return true;
    for(let i=0;i<channels[0].length;i++){let x=0;for(const ch of channels)x+=ch[i];x/=channels.length;this.ring[this.cursor]=x;let filtered=0;for(let k=0;k<63;k++)filtered+=this.filter[k]*this.ring[(this.cursor-k+63)%63];this.cursor=(this.cursor+1)%63;
      while(this.next<=this.index){const t=Math.max(0,Math.min(1,this.next-(this.index-1))),y=this.prev+(filtered-this.prev)*t;this.pcm[this.used++]=Math.round(Math.max(-1,Math.min(1,y))*32767);this.energy+=y*y;this.next+=this.step;
        if(this.used===1600){const buffer=new ArrayBuffer(3200),view=new DataView(buffer);for(let j=0;j<1600;j++)view.setInt16(j*2,this.pcm[j],true);this.port.postMessage({pcm:buffer,level:Math.sqrt(this.energy/this.used)},[buffer]);this.used=0;this.energy=0;}}
      this.prev=filtered;this.index++;}return true;}}
registerProcessor('pcm-processor',PCMProcessor);
