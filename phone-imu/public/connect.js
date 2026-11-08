import {pairingToken} from './shared.js';
const $=id=>document.getElementById(id);
async function pair(token){
  $('error').textContent='';$('links').hidden=true;
  try{
    const response=await fetch('/api/links',{headers:{Authorization:`Bearer ${token}`}});
    if(!response.ok)throw new Error('Pairing key not accepted. Check it and try again.');
    const links=await response.json();localStorage.setItem('imu-token',token);
    $('phone').href=links.phoneURL;$('links').hidden=false;$('pair').hidden=true;
  }catch(error){$('pair').hidden=false;$('error').textContent=error.message;}
}
$('pair').onsubmit=event=>{event.preventDefault();pair($('key').value.trim());};
const token=pairingToken();if(token)pair(token);
