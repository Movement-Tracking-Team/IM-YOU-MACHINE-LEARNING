export function cloudConfig(env){
  if(!env.IMU_TOKEN||env.IMU_TOKEN.length<32)throw new Error('Set IMU_TOKEN to a random secret of at least 32 characters');
  if(!env.IMU_RECORDINGS)throw new Error('Set IMU_RECORDINGS to a persistent storage directory');
  return {token:env.IMU_TOKEN};
}
