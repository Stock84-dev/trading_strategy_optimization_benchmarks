use crate::context::Pair;
use crate::{Account, Context, RsiState};
use mouse::prelude::*;
use nebuchadnezzar::core::serde;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

const MIN_TIMEFRAME: u32 = 60;
const TIMEFRAME_RANGE: u32 = 60;
const TIMEFRAME_STEP: u32 = 60;
const MAX_TIMEFRAME: u32 = MIN_TIMEFRAME + TIMEFRAME_RANGE * TIMEFRAME_STEP;
const MAX_LEN: u32 = 100;
const MAX_HLINE: u32 = 101;
const MAX_LLINE: u32 = 101;

pub struct State {
    pub timeframe: u32,
    pub len: u32,
    pub elapsed_ns: u128,
}

impl Default for State {
    fn default() -> Self {
        State {
            timeframe: MIN_TIMEFRAME,
            len: 2,
            elapsed_ns: 0,
        }
    }
}

pub struct Mapper {
    state: State,
    space_file: Mutex<File>,
    state_file: File,
}

impl Mapper {
    pub fn read_top_10(&mut self) -> Vec<Pair> {
        let mut file = self.space_file.lock().unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        let mut results: &mut [AccountResult] =
            unsafe { ptr_as_slice_mut(data.as_mut_ptr(), data.len() / AccountResult::size()) };
        let mut top10 = vec![Pair::default(); 10];
        println!("{}", results.len());
        //        let mut pairs = vec![];
        for (i, result) in results.into_iter().enumerate() {
            let timeframe = result.timeframe;
            let len = result.len;
            let hline = result.hline;
            let lline = result.lline;
            if timeframe == 3540 && len == 19 && hline == 22 && lline == 40 {
                dbg!(&result);
            }
            //            pairs.push(Pair {
            //                params: [timeframe, len, hline, lline],
            //                balance: result.balance,
            //            });

            //            println!("{} {} {} {}", timeframe, len, hline, lline);

            let (max_i, pair) = top10
                .iter()
                .enumerate()
                .min_by_key(|x| OrderedFloat(x.1.balance))
                .unwrap();
            if result.balance > pair.balance {
                top10[max_i] = Pair {
                    params: [timeframe, len, hline, lline],
                    balance: result.balance,
                    account: Default::default(),
                };
            }
        }
        //        pairs.sort_by(|a, b| {
        //            OrderedFloat(b.balance)
        //                .partial_cmp(&OrderedFloat(a.balance))
        //                .unwrap()
        //        });
        top10
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn new() -> Result<Self> {
        unsafe {
            let state_path = Path::new("./state.bin");
            let exists = state_path.exists();
            let mut state_file = OpenOptions::new()
                .create(true)
                .write(true)
                .read(true)
                .open(state_path)?;

            let mut state = State::default();
            if exists {
                state_file.read_exact(state.as_u8_slice_mut())?;
            }
            let mut space_file = OpenOptions::new()
                .write(true)
                .read(true)
                .create(true)
                .open("./search_space.bin")?;
            space_file.seek(SeekFrom::End(0))?;
            Ok(Self {
                state,
                space_file: Mutex::new(space_file),
                state_file,
            })
        }
    }

    pub async fn map(&mut self) -> Result<()> {
        let context = Context::new(10).await?;
        let mut elapsed_timeframe_ns = 0;
        let mut elapsed_cache_ns = 0;
        let mut elapsed_compute_ns = 0;

        for timeframe in (self.state.timeframe..MAX_TIMEFRAME).step_by(TIMEFRAME_STEP as usize) {
            let now = Instant::now();
            let hlcvs = context.build_hlcvs(timeframe);
            elapsed_timeframe_ns += now.elapsed().as_nanos();
            for len in self.state.len as usize..MAX_LEN as usize {
                let now = Instant::now();
                let mut cache = Vec::with_capacity(hlcvs.len());
                let mut state = RsiState::new(len, &hlcvs);
                let mut offset = state.update_at();
                for _ in 0..offset {
                    cache.push(0.0f32);
                }
                for i in offset..hlcvs.len() {
                    cache.push(state.update(&hlcvs, i));
                }
                elapsed_cache_ns += now.elapsed().as_nanos();
                offset += 1;
                let now = Instant::now();
                (0..MAX_HLINE).into_par_iter().for_each(|hline| {
                    let mut accounts = vec![Account::new(); MAX_LLINE as usize];
                    let mut prev_close = hlcvs[offset - 1].close;
                    let mut prev_rsi = cache[offset - 1];
                    let hline = hline as f32;
                    for i in offset..hlcvs.len() {
                        let rsi = cache[i];
                        for lline in 0..MAX_LLINE as usize {
                            let account = unsafe { accounts.get_unchecked_mut(lline) };
                            account.update(
                                &hlcvs,
                                i,
                                prev_close,
                                prev_rsi,
                                rsi,
                                context.max_risk,
                                hline,
                                lline as f32,
                            );
                        }
                        prev_close = hlcvs[i].close;
                        prev_rsi = rsi;
                    }
                    let results: Vec<_> = accounts
                        .into_iter()
                        .enumerate()
                        .map(|(i, x)| AccountResult {
                            timeframe,
                            len: len as u32,
                            hline: hline as u32,
                            lline: i as u32,
                            balance: x.balance,
                            max_balance: x.max_balance,
                            max_drawdown: x.max_drawdown,
                            n_trades: x.n_trades,
                        })
                        .collect();
                    let mut file = self.space_file.lock().unwrap();
                    let results = unsafe {
                        ptr_as_slice(results.as_ptr(), AccountResult::size() * results.len())
                    };
                    file.write_all(results).unwrap();
                });
                self.state.timeframe = timeframe;
                self.state.len = len as u32;
                self.state.elapsed_ns += now.elapsed().as_nanos();
                self.state_file.seek(SeekFrom::Start(0))?;
                self.state_file.write_all(self.state.as_u8_slice())?;
                println!(
                    "{} comb/s",
                    MAX_HLINE as f32 * MAX_LLINE as f32 / now.elapsed().as_nanos() as f32 * 1e9
                );
                println!(
                    "{}/{}",
                    (self.state.timeframe - MIN_TIMEFRAME) / TIMEFRAME_STEP * (MAX_LEN - 2)
                        + self.state.len,
                    TIMEFRAME_RANGE * (MAX_LEN - 2)
                );
            }
            self.state.len = 2;
            self.state_file.seek(SeekFrom::Start(0))?;
            self.state_file.write_all(self.state.as_u8_slice())?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AccountResult {
    pub timeframe: u32,
    pub len: u32,
    pub hline: u32,
    pub lline: u32,
    pub balance: f32,
    pub max_balance: f32,
    pub max_drawdown: f32,
    pub n_trades: u32,
}
