use crate::{load_hlcv, plot, Account, RsiState};
use cmaes::{CMAESOptions, ParallelObjectiveFunction, PlotOptions};
use cmaes::{DVector, Mode, ObjectiveFunction};
use half::f16;
use merovingian::hlcv::{Hlcv, MappedHlcvs};
use mouse::prelude::*;
use mouse::sync::{Mutex, RwLock, RwLockReadGuard};
use ndarray::ArrayView1;
use num_traits::Float;
use optimize::NelderMeadBuilder;
use ordered_float::OrderedFloat;
use pos_pso::SwarmConfig;
use rand::prelude::*;
use simplers_optimization::Optimizer;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

static mut ELAPSED_NS: u64 = 0;

const MIN_TIMEFRAME: u32 = 60;
const TIMEFRAME_RANGE: u32 = 60;
const TIMEFRAME_STEP: u32 = 60;
const MAX_TIMEFRAME: u32 = MIN_TIMEFRAME + TIMEFRAME_RANGE * TIMEFRAME_STEP;
const MAX_LEN: u32 = 100;
const MAX_HLINE: u32 = 101;
const MAX_LLINE: u32 = 101;

pub struct Context {
    pub time_taken_ns: AtomicU64,
    pub timeframe_cache: RwLock<HashMap<u32, Arc<Vec<Hlcv>>>>,
    pub timeframe: AtomicU32,
    pub hybrid: bool,
    pub compute_cost: AtomicU64,
    pub cache_cost: AtomicU64,
    pub timeframe_cost: AtomicU64,
    pub cache_timeframe: bool,
    pub hlcvs_start_ts: u32,
    pub max_risk: f32,
    pub micros_compute: AtomicU64,
    pub micros_cache: AtomicU64,
    pub micros_timeframe: AtomicU64,
    pub n_shards: usize,
    pub shard_i: usize,
    pub histogram: Mutex<Vec<u32>>,
    pub bucket_size: f32,
    pub params: Mutex<HashMap<u32, HashSet<usize>>>,
    // top 10 najboljih parametra
    pub top: Mutex<Vec<Pair>>,
}

impl Context {
    pub async fn new(n_accounts: usize) -> Result<Self> {
        let start_ts = 1443184345;
        let count = 1633869265 - 1443184345;
        let count = 190684900 * 3 / 4;
        // let start_ts = start_ts + count as u32;
        // let count = 1_000 * 60 * 60 * 2;
        // let count = 1_000 * 60 * 60 * 2;
        // let count = 10000000;
        //        let count = count / 1000;
        let now = Instant::now();
        let hlcvs_mapped = load_hlcv("BitMEX", "XBTUSD", start_ts, count).await?;
        let mut cache = HashMap::new();
        let mut hlcvs = Vec::new();
        hlcvs.extend_from_slice(hlcvs_mapped.as_ref());
        println!("{}ms", now.elapsed().as_millis());
        // std::process::exit(0);
        cache.insert(1, Arc::new(hlcvs));
        Ok(Self {
            hlcvs_start_ts: hlcvs_mapped.start_ts,
            max_risk: 0.01,
            timeframe: Default::default(),
            hybrid: false,
            micros_compute: Default::default(),
            micros_cache: Default::default(),
            micros_timeframe: Default::default(),
            n_shards: 0,
            timeframe_cache: RwLock::new(cache),
            compute_cost: Default::default(),
            cache_cost: Default::default(),
            timeframe_cost: Default::default(),
            cache_timeframe: false,
            top: Mutex::new(vec![Default::default(); n_accounts]),
            shard_i: 0,
            time_taken_ns: AtomicU64::default(),
            histogram: Mutex::default(),
            params: Default::default(),
            bucket_size: 0.1,
        })
    }

    // pub fn clear() {
    //         micros_compute: Default::default(),
    //         micros_cache: Default::default(),
    //         micros_timeframe: Default::default(),
    // }

    fn get_hlcvs<'a>(&'a self, timeframe: u32) -> Arc<Vec<Hlcv>> {
        if timeframe == 1 {
            return self.timeframe_cache.read()[&1].clone();
        }
        if self.cache_timeframe {
            let guard = self.timeframe_cache.read();
            if guard.contains_key(&timeframe) {
                return guard[&timeframe].clone();
            }
            drop(guard);
            let hlcvs = self.build_hlcvs(timeframe);
            let mut guard = self.timeframe_cache.write();
            guard.insert(timeframe, Arc::new(hlcvs));
            return guard[&timeframe].clone();
        }
        let hlcvs = self.build_hlcvs(timeframe);
        let mut guard = self.timeframe_cache.write();
        guard.insert(0, Arc::new(hlcvs));
        return guard[&0].clone();
    }

    pub fn build_hlcvs(&self, timeframe: u32) -> Vec<Hlcv> {
        let mut changed_hlcvs = vec![];
        let guard = self.timeframe_cache.read();
        let src = &guard[&1][..];
        let now = Instant::now();
        changed_hlcvs = vec![
            Default::default();
            merovingian::hlcv::change_timeframe_dest_len(
                src.len(),
                self.hlcvs_start_ts,
                1,
                timeframe,
            )
        ];
        merovingian::hlcv::change_timeframe(
            &src,
            self.hlcvs_start_ts,
            1,
            timeframe,
            &mut changed_hlcvs,
        );
        self.timeframe_cost
            .fetch_add(src.len() as u64, Ordering::SeqCst);
        self.micros_timeframe
            .fetch_add(now.elapsed().as_micros() as u64, Ordering::SeqCst);
        changed_hlcvs
    }

    pub fn backtest16(&self, timeframe: u32, period: usize, hline: f32, lline: f32) -> Account {
        //        println!("{}", timeframe);
        let mut nanos_cache = 0u128;
        let mut nanos_compute = 0u128;
        let hlcvs = self.get_hlcvs(timeframe);
        let low = hlcvs
            .iter()
            .map(|x| x.low)
            .reduce(|acc, x| acc.min(x))
            .unwrap();
        let hlcvs = hlcvs
            .iter()
            .map(|x| Hlcv {
                high: x.high / low,
                low: x.low / low,
                close: x.close / low,
                volume: x.volume,
            })
            .collect::<Vec<_>>();
        let hlcvs = hlcvs
            .iter()
            .map(|x| crate::account16::Hlcv {
                high: f16::from_f32(x.high),
                low: f16::from_f32(x.low),
                close: f16::from_f32(x.close),
                volume: f16::from_f32(x.volume),
            })
            .collect::<Vec<_>>();
        let hlcvs = &hlcvs[..];
        let mut now = Instant::now();
        let mut state = crate::account16::RsiState::new(period, &hlcvs);
        let mut offset = state.update_at();
        let mut prev_rsi = state.update(hlcvs, offset);
        nanos_cache += now.elapsed().as_nanos();
        let mut prev_close = hlcvs[offset].close;
        offset += 1;
        let mut account = crate::account16::Account16::new();
        for i in offset..hlcvs.len() {
            let mut now = Instant::now();
            let rsi = state.update(hlcvs, i);
            nanos_cache += now.elapsed().as_nanos();
            let mut now = Instant::now();
            account.update(
                &hlcvs,
                i,
                prev_close,
                prev_rsi,
                rsi,
                f16::from_f32(self.max_risk),
                f16::from_f32(hline),
                f16::from_f32(lline),
            );
            prev_close = hlcvs[i].close;
            prev_rsi = rsi;
            nanos_compute += now.elapsed().as_nanos();
        }
        dbg!(&account);
        println!("{}", now.elapsed().as_micros());
        self.cache_cost
            .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
        self.compute_cost
            .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
        self.micros_cache
            .fetch_add((nanos_cache / 1000) as u64, Ordering::SeqCst);
        self.micros_compute
            .fetch_add((nanos_compute / 1000) as u64, Ordering::SeqCst);
        // self.rank(
        //     timeframe,
        //     period as u32,
        //     hline as u32,
        //     lline as u32,
        //     &account,
        // );
        // account
        panic!();
    }

    pub fn backtest_plot(&self, timeframe: u32, period: usize, hline: f32, lline: f32) -> Account {
        let mut balances = Vec::new();
        {
            let mut params = self.params.lock();
            params
                .entry(timeframe)
                .and_modify(|x| {
                    x.insert(period);
                })
                .or_insert_with(|| {
                    let mut set = HashSet::new();
                    set.insert(period);
                    set
                });
        }
        //        println!("{}", timeframe);
        let mut nanos_cache = 0u128;
        let mut nanos_compute = 0u128;
        let hlcvs = self.get_hlcvs(timeframe);
        // get lowest low of all hlcvs
        // let low = hlcvs.iter().map(|x| x.low).reduce(f32::min).unwrap();
        // let hlcvs = hlcvs
        //     .iter()
        //     .map(|x| Hlcv {
        //         high: x.high / low,
        //         low: x.low / low,
        //         close: x.close / low,
        //         volume: x.volume,
        //     })
        //     .collect::<Vec<_>>();
        let hlcvs = &hlcvs[..];
        let mut now = Instant::now();
        let mut state = RsiState::new(period, &hlcvs);
        let mut offset = state.update_at();
        let mut prev_rsi = state.update(hlcvs, offset);
        nanos_cache += now.elapsed().as_nanos();
        let mut prev_close = hlcvs[offset].close;
        offset += 1;
        let mut account = Account::new();
        for i in offset..hlcvs.len() {
            let mut now = Instant::now();
            let rsi = state.update(hlcvs, i);
            nanos_cache += now.elapsed().as_nanos();
            let mut now = Instant::now();
            account.update(
                &hlcvs,
                i,
                prev_close,
                prev_rsi,
                rsi,
                self.max_risk,
                hline,
                lline,
            );
            prev_close = hlcvs[i].close;
            prev_rsi = rsi;
            nanos_compute += now.elapsed().as_nanos();
            balances.push(account.balance);
        }
        if account.n_trades < 30 {
            account.balance = 0.;
        }
        // dbg!(&account);
        // println!("{}", now.elapsed().as_micros());
        self.cache_cost
            .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
        self.compute_cost
            .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
        self.micros_cache
            .fetch_add((nanos_cache / 1000) as u64, Ordering::SeqCst);
        self.micros_compute
            .fetch_add((nanos_compute / 1000) as u64, Ordering::SeqCst);
        self.rank(
            timeframe,
            period as u32,
            hline as u32,
            lline as u32,
            &account,
        );
        crate::plot::plot_values("equity_curve", &balances).unwrap();
        account
    }

    pub fn backtest(&self, timeframe: u32, period: usize, hline: f32, lline: f32) -> Account {
        {
            let mut params = self.params.lock();
            params
                .entry(timeframe)
                .and_modify(|x| {
                    x.insert(period);
                })
                .or_insert_with(|| {
                    let mut set = HashSet::new();
                    set.insert(period);
                    set
                });
        }
        //        println!("{}", timeframe);
        let mut nanos_cache = 0u128;
        let mut nanos_compute = 0u128;
        let hlcvs = self.get_hlcvs(timeframe);
        // get lowest low of all hlcvs
        // let low = hlcvs.iter().map(|x| x.low).reduce(f32::min).unwrap();
        // let hlcvs = hlcvs
        //     .iter()
        //     .map(|x| Hlcv {
        //         high: x.high / low,
        //         low: x.low / low,
        //         close: x.close / low,
        //         volume: x.volume,
        //     })
        //     .collect::<Vec<_>>();
        let hlcvs = &hlcvs[..];
        let mut now = Instant::now();
        let mut state = RsiState::new(period, &hlcvs);
        let mut offset = state.update_at();
        let mut prev_rsi = state.update(hlcvs, offset);
        nanos_cache += now.elapsed().as_nanos();
        let mut prev_close = hlcvs[offset].close;
        offset += 1;
        let mut account = Account::new();
        for i in offset..hlcvs.len() {
            let mut now = Instant::now();
            let rsi = state.update(hlcvs, i);
            nanos_cache += now.elapsed().as_nanos();
            let mut now = Instant::now();
            account.update(
                &hlcvs,
                i,
                prev_close,
                prev_rsi,
                rsi,
                self.max_risk,
                hline,
                lline,
            );
            prev_close = hlcvs[i].close;
            prev_rsi = rsi;
            nanos_compute += now.elapsed().as_nanos();
        }
        if account.n_trades < 30 {
            account.balance = 0.;
        }
        // dbg!(&account);
        // println!("{}", now.elapsed().as_micros());
        self.cache_cost
            .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
        self.compute_cost
            .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
        self.micros_cache
            .fetch_add((nanos_cache / 1000) as u64, Ordering::SeqCst);
        self.micros_compute
            .fetch_add((nanos_compute / 1000) as u64, Ordering::SeqCst);
        self.rank(
            timeframe,
            period as u32,
            hline as u32,
            lline as u32,
            &account,
        );
        account
    }

    pub fn random_search(&mut self, n_samples: usize) {
        let now = Instant::now();
        let i = AtomicI32::new(0);
        (0..n_samples).into_iter().for_each(|_| {
            let timeframe = random::<u32>() % TIMEFRAME_RANGE * TIMEFRAME_STEP + MIN_TIMEFRAME;
            let len = random::<usize>() % MAX_LEN as usize;
            let hline = random::<u32>() % MAX_HLINE;
            let lline = random::<u32>() % MAX_LLINE;
            let hline = hline as f32;
            let lline = lline as f32;
            self.backtest(timeframe, len, hline, lline);
            let i = i.fetch_add(1, Ordering::SeqCst) + 1;
            // println!(
            //     "{}/{}, {} combs/s",
            //     i,
            //     end,
            //     i as f32 / (now.elapsed().as_nanos() as f32 / 1e9)
            // );
        });
    }

    //[sparring/src/main.rs:44] top10.iter().map(|x|
    //                x.balance).sorted_by_key(|x| OrderedFloat(*x)).collect_vec() = [
    //    128.72197,
    //    132.06981,
    //    133.41286,
    //    135.42238,
    //    143.17303,
    //    150.56244,
    //    152.15628,
    //    169.24411,
    //    171.70618,
    //    260.32178,
    //]
    //11634.352 - secs taken
    pub fn grid_search(&mut self) {
        let now = Instant::now();
        let total = TIMEFRAME_RANGE as u64 * MAX_LEN as u64 * MAX_HLINE as u64 * MAX_LLINE as u64;
        for timeframe in (MIN_TIMEFRAME..MAX_TIMEFRAME).step_by(TIMEFRAME_STEP as usize) {
            let hlcvs = self.build_hlcvs(timeframe);
            for len in 2..MAX_LEN as usize {
                let mut cache = Vec::with_capacity(hlcvs.len());
                let mut state = RsiState::new(len, &hlcvs);
                let mut offset = state.update_at();
                for _ in 0..offset {
                    cache.push(0.0f32);
                }
                for i in offset..hlcvs.len() {
                    cache.push(state.update(&hlcvs, i));
                }
                self.cache_cost
                    .fetch_add(hlcvs.len() as u64, Ordering::SeqCst);
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
                                self.max_risk,
                                hline,
                                lline as f32,
                            );
                        }
                        prev_close = hlcvs[i].close;
                        prev_rsi = rsi;
                    }
                    for i in 0..MAX_LLINE {
                        // if accounts[i as usize].balance > 1. {
                        //     println!("{}", accounts[i as usize].balance);
                        // }
                        if accounts[i as usize].n_trades < 30 {
                            accounts[i as usize].balance = 0.;
                        }
                        self.rank(
                            timeframe,
                            len as u32,
                            hline as u32,
                            i,
                            &accounts[i as usize],
                        );
                    }
                    self.compute_cost
                        .fetch_add(hlcvs.len() as u64 * MAX_LLINE as u64, Ordering::SeqCst);
                });
                plot::save_hist("grid", &self.histogram.lock(), self.bucket_size).unwrap();
                plot::save_pairs("grid", &mut *self.top.lock()).unwrap();
                // let values = self
                //     .histogram
                //     .lock()
                //     .iter()
                //     .enumerate()
                //     .skip(2)
                //     .map(|x| (x.0 as f32 * 0.5, *x.1 as f32))
                //     .collect::<Vec<_>>();
                // plot::plot_xy("grid hist", &values).unwrap();
                // println!(
                //     "{} comb/s",
                //     MAX_HLINE as f32 * MAX_LLINE as f32 / now.elapsed().as_nanos() as f32 * 1e9
                // )
            }
        }
        self.micros_compute
            .store(now.elapsed().as_micros() as u64, Ordering::SeqCst);
    }

    pub fn genetic_search(&self) {}

    pub fn genetic_search_param(&self, pop: usize, gen: usize, cm: f64) {
        let mut param_count = 4;
        // Ako se koristi GA + BO algoritam
        if self.hybrid {
            param_count = 3;
        }
        let first_guess = vec![0.5; param_count];
        // Konfiguracija broja populacije i generacije
        // let pop = 1000_000;
        // let gen = 1000;
        // println!("{pop}pop {gen}gen");

        // Konfiguriranje CMA-ES algoritma.
        let mut cmaes_state = CMAESOptions::new(first_guess, 0.5)
            .mode(Mode::Maximize)
            // .seed(0)
            .population_size(pop)
            .max_generations(gen)
            // .tol_fun(0.)
            // .tol_fun_rel(0.)
            // .tol_fun_hist(0.)
            // .tol_x(0.)
            // .tol_stagnation(usize::MAX)
            // .tol_x_up(f64::MAX)
            // .tol_condition_cov(f64::MAX)
            // .cm(cm)
            // "self" (Context struktura) ima implementiran "trait" koji sadrži funkciju koja se
            // pokreće za svaku jedinku.
            .build(self)
            // Terminiranje programa ukoliko se pojavi neka greška.
            .unwrap();

        // Pokretanje optimizacije.
        cmaes_state.run();
    }

    // 567 in 30 iters, 477.86502s
    pub fn pso(&self) {
        extern crate pos_pso;
        use pos_pso::{JobConfig, PSOConfig, PSO};
        let me: &'static Self = unsafe { std::mem::transmute(self) };

        let f = |x: &[f64]| {
            let now = Instant::now();
            let timeframe = MIN_TIMEFRAME + TIMEFRAME_STEP * x[0] as u32;
            let len = x[1] as usize;
            let hline = x[2] as u32 as f32;
            let lline = x[3] as u32 as f32;
            let account = me.backtest(timeframe, len, hline, lline);
            unsafe {
                ELAPSED_NS += now.elapsed().as_nanos() as u64;
            }
            1. / account.balance as f64
        };

        // Create a PSO Configuration:
        let pso_config = PSOConfig::new(
            1,    // 1 swarm used in optimization
            100,  // 256 particles are spawned
            1,    // console is updated every 10 itterations
            true, // optimizer is verbose (will provide more detailed information to console)
        );

        // Create a PSO:
        let pso = PSO::new(pso_config);
        let num_variables = 4;

        // Create a Job Configuration:
        let job_config = JobConfig::new(
            num_variables,
            vec![
                [0., TIMEFRAME_RANGE as f64],
                [2., MAX_LEN as f64],
                [0., MAX_HLINE as f64],
                [0., MAX_LLINE as f64],
            ], // [upper, lower] bounds for each variable
            vec![1.125; num_variables], // max velocity for each variable
            10,                         // run for 100 itterations
            0.0000001, // exit cost (optimization will stop when a cost of 0.0000001 is reached)
        );

        // Minimize cost function:

        //use minimize_independant to optimize with the default independant-swarm configuration
        //the next example will show how to use collaborative-swarms
        let min = pso.minimize_independant(job_config, f);

        println!("Minimum of: {}, With value: {:?}", min.0, min.1);
    }

    // compute: 727.04126s
    // [1740, 42, 25, 41]: 421.93964
    pub fn pso2(&self) {
        extern crate pos_pso;
        use pos_pso::{JobConfig, PSOConfig, PSO};
        let me: &'static Self = unsafe { std::mem::transmute(self) };

        let f = |x: &[f64]| {
            let now = Instant::now();
            let timeframe = MIN_TIMEFRAME + TIMEFRAME_STEP * x[0] as u32;
            let len = x[1] as usize;
            let hline = x[2] as u32 as f32;
            let lline = x[3] as u32 as f32;
            let account = me.backtest(timeframe, len, hline, lline);
            unsafe {
                ELAPSED_NS += now.elapsed().as_nanos() as u64;
            }
            1. / account.balance as f64
        };

        // Create a PSO Configuration:
        let pso_config = PSOConfig::new(
            4,    // 1 swarm used in optimization
            25,   // 256 particles are spawned
            1,    // console is updated every 10 itterations
            true, // optimizer is verbose (will provide more detailed information to console)
        );

        // Create a PSO:
        let pso = PSO::new(pso_config);
        let num_variables = 4;

        // Create a Job Configuration:
        let job_config = JobConfig::new(
            num_variables,
            vec![
                [0., TIMEFRAME_RANGE as f64],
                [2., MAX_LEN as f64],
                [0., MAX_HLINE as f64],
                [0., MAX_LLINE as f64],
            ], // [upper, lower] bounds for each variable
            vec![1.125; num_variables], // max velocity for each variable
            100,                        // run for 100 itterations
            0.0000001, // exit cost (optimization will stop when a cost of 0.0000001 is reached)
        );

        let swarm_config = SwarmConfig::new_collaborative(
            1.45,  // local weigth:    how much particles care about their best known location
            1.6, // tribal weight:   how much particles care about their swarms best known location
            1.25, // global weight:   how much particles care about the overall best known location
            0.4, // inertial coefficient:    component of a particles velocity that contributes to its next velocity
            1.25, // inertial growth factor:  how much inertia grows and shrinks throughout optimization
            0.125, // wall bounce factor:      component of velocity that is saved when particle goes out of bounds
            10, // tribal-global collab period:   swarms share best known location every 10 itterations
        );

        // Minimize cost function:

        //use minimize_independant to optimize with the default independant-swarm configuration
        //the next example will show how to use collaborative-swarms
        let min = pso.minimize(job_config, swarm_config, f);

        println!("Minimum of: {}, With value: {:?}", min.0, min.1);
    }

    pub fn abc(&self) {
        extern crate abc;
        extern crate rand;

        use abc::{Candidate, Context, HiveBuilder};
        use rand::{random, thread_rng, Rng};
        use std::f32::consts::PI;

        const SIZE: usize = 4;

        #[derive(Clone, Debug)]
        struct S([f32; SIZE]);

        // Not really necessary; we're using this mostly to demonstrate usage.
        struct SBuilder {
            min: f32,
            max: f32,
            a: f32,
            p_min: f32,
            p_max: f32,
            context: &'static self::Context,
        }

        impl Context for SBuilder {
            type Solution = [f32; SIZE];

            fn make(&self) -> [f32; SIZE] {
                let timeframe = random::<u32>() % TIMEFRAME_RANGE * TIMEFRAME_STEP + MIN_TIMEFRAME;
                let len = random::<usize>() % MAX_LEN as usize;
                let hline = random::<u32>() % MAX_HLINE;
                let lline = random::<u32>() % MAX_LLINE;
                [timeframe as f32, len as f32, hline as f32, lline as f32]
            }

            fn evaluate_fitness(&self, solution: &[f32; 4]) -> f64 {
                self.context
                    .backtest(
                        solution[0].clamp(MIN_TIMEFRAME as f32, MAX_TIMEFRAME as f32) as u32,
                        solution[1].clamp(2., MAX_LEN as f32) as usize,
                        solution[2].clamp(0., 100.),
                        solution[3].clamp(0., 100.),
                    )
                    .balance as f64
            }

            fn explore(&self, field: &[Candidate<[f32; SIZE]>], index: usize) -> [f32; SIZE] {
                let ref current = field[index].solution;
                let mut new = [0_f32; SIZE];

                for i in 0..SIZE {
                    // Choose a different vector at random.
                    let mut rng = thread_rng();
                    let mut index2 = rng.gen_range(0..current.len() - 1);
                    if index2 >= index {
                        index2 += 1;
                    }
                    let ref other = field[index2].solution;

                    let phi = random::<f32>() * (self.p_max - self.p_min) + self.p_min;
                    new[i] = current[i] + (phi * (current[i] - other[i]));
                }

                new
            }
        }

        let mut builder = SBuilder {
            min: -5.12,
            max: 5.12,
            a: 10.0,
            p_min: -1.0,
            p_max: 1.0,
            context: unsafe { std::mem::transmute(self) },
        };
        let hive_builder = HiveBuilder::new(builder, 10);
        let hive = hive_builder.build().unwrap();
        for new_best in hive.stream().iter().take(10) {
            println!("{}", new_best.fitness);
        }
    }

    pub fn de(&self) {
        extern crate differential_evolution;

        use differential_evolution::self_adaptive_de;

        // create a self adaptive DE with an inital search area
        // from -10 to 10 in 5 dimensions.

        let search_space = vec![
            (0., TIMEFRAME_RANGE as f32),
            (2., MAX_LEN as f32),
            (0., MAX_HLINE as f32),
            (0., MAX_LLINE as f32),
        ];
        let mut settings = differential_evolution::Settings::default(search_space, |pos| {
            self.backtest(
                pos[0].clamp(MIN_TIMEFRAME as f32, MAX_TIMEFRAME as f32) as u32,
                pos[1].clamp(2., MAX_LEN as f32) as usize,
                pos[2].clamp(0., 100.) as u32 as f32,
                pos[3].clamp(0., 100.) as u32 as f32,
            )
            .balance
        });
        let pop_size = 1000;
        settings.pop_size = pop_size;
        let mut population = differential_evolution::Population::new(settings);

        // perform 10000 cost evaluations
        for x in population.iter().step_by(pop_size).take(100) {
            self.summary();
        }

        // show the result
        let (cost, pos) = population.best().unwrap();
        println!("cost: {}", cost);
        println!("pos: {:?}", pos);
    }

    // pub fn sa(&self) {
    //     // use rand::SeedableRng;
    //     use simulated_annealing::{Bounds, NeighbourMethod, Point, Schedule, Status, APF, SA};
    //
    //     // Define the objective function
    //     let f = |p: &Point<f32, 4>| -> Result<f32> {
    //         Ok(1.
    //             / self
    //                 .backtest(
    //                     MIN_TIMEFRAME
    //                         + TIMEFRAME_STEP
    //                         + p[0].clamp(1., TIMEFRAME_RANGE as f32) as u32,
    //                     p[1].clamp(2., MAX_LEN as f32) as usize,
    //                     p[2].clamp(0., 100.) as u32 as f32,
    //                     p[3].clamp(0., 100.) as u32 as f32,
    //                 )
    //                 .balance)
    //     };
    //     let mut rng = rand::rngs::SmallRng::from_entropy();
    //     // Get the minimum (and the corresponding point)
    //     let (m, p) = SA {
    //         // Objective function
    //         f,
    //         // Initial point
    //         p_0: &[30., 50., 50., 50.],
    //         // Initial temperature
    //         t_0: 100_00.0,
    //         // Minimum temperature
    //         t_min: 1.0,
    //         // Bounds of the parameter space
    //         bounds: &[
    //             0. ..TIMEFRAME_RANGE as f32,
    //             2. ..MAX_LEN as f32,
    //             0. ..MAX_HLINE as f32,
    //             0. ..MAX_LLINE as f32,
    //         ],
    //         // Acceptance probability function
    //         apf: &APF::Metropolis,
    //         // Method of getting a random neighbour
    //         neighbour: &NeighbourMethod::Normal { sd: 5. },
    //         // Annealing schedule
    //         schedule: &Schedule::Fast,
    //         // Status function
    //         status: &mut Status::Periodic { nk: 10 },
    //         // Random number generator
    //         rng: &mut rng,
    //     }
    //     .findmin()
    //     .unwrap();
    //     dbg!(m, p);
    // }

    pub fn spsa(&self) {
        use spsa::{maximize, Optimizer, Options};

        let mut optimizer = Optimizer::new();
        let mut input = [30.0, 25.0, 30., 70.];

        optimizer.optimize(
            maximize(|data| {
                dbg!(&data);
                let timeframe = MIN_TIMEFRAME + TIMEFRAME_STEP * data[0] as u32;
                let len = data[1] as usize;
                let hline = data[2] as u32 as f32;
                let lline = data[3] as u32 as f32;
                let account = self.backtest(timeframe, len, hline, lline);
                account.balance as f64
            }),
            &mut input,
            Options {
                iterations: 100,
                ..Default::default()
            },
        );

        dbg!(input);
    }

    pub fn tpe(&self) {
        use rand::SeedableRng as _;
        let opt = |start, end| {
            tpe::TpeOptimizerBuilder::new()
                .gamma(0.003125)
                // .gamma(0.05)
                .candidates(32)
                .build(tpe::parzen_estimator(), tpe::range(start, end).unwrap())
                .unwrap()
        };

        let mut best_value = std::f64::INFINITY;
        for i in 0..1 {
            let mut timeframe_opt = opt(1., 60.);
            let mut period_opt = opt(2., 100.);
            let mut hline_opt = opt(0., 100.);
            let mut lline_opt = opt(0., 100.);
            let mut rng = SmallRng::from_entropy();
            let n_parallel = 1;
            // TODO: we could run GA and save results then later on we use those results to jump
            // start TPE
            for _ in 0..8000 {
                let mut timeframes = Vec::new();
                let mut periods = Vec::new();
                let mut llines = Vec::new();
                let mut hlines = Vec::new();
                let mut tells = Vec::new();
                for _ in 0..n_parallel {
                    let provided_timeframe = timeframe_opt
                        .ask(&mut rng)
                        .unwrap()
                        .clamp(1., TIMEFRAME_RANGE as f64);
                    let timeframe = MIN_TIMEFRAME + TIMEFRAME_STEP * provided_timeframe as u32;
                    let period =
                        period_opt.ask(&mut rng).unwrap().clamp(2., MAX_LEN as f64) as usize;
                    let hline = hline_opt.ask(&mut rng).unwrap().clamp(0., 100.) as u32 as f32;
                    let lline = lline_opt.ask(&mut rng).unwrap().clamp(0., 100.) as u32 as f32;
                    timeframes.push(provided_timeframe);
                    periods.push(period);
                    hlines.push(hline);
                    llines.push(lline);
                    let account = self.backtest(timeframe as u32, period, hline, lline);
                    let balance = 1. / account.balance as f64;
                    dbg!(account.balance);
                    best_value = best_value.min(balance);
                    tells.push(balance);
                }
                for i in 0..n_parallel {
                    let provided_timeframe = timeframes[i];
                    let period = periods[i];
                    let hline = hlines[i];
                    let lline = llines[i];
                    let balance = tells[i];
                    timeframe_opt
                        .tell(provided_timeframe as f64, balance)
                        .unwrap();
                    period_opt.tell(period as f64, balance).unwrap();
                    hline_opt.tell(hline as f64, balance).unwrap();
                    lline_opt.tell(lline as f64, balance).unwrap();
                }
            }
            dbg!(1. / best_value);
        }
    }

    pub fn nelder_mead(&self) {
        use ndarray::Array;
        use optimize::Minimizer;
        // Define a function that we aim to minimize
        let function =
            |x: ArrayView1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);

        // Create a minimizer using the builder pattern. If some of the parameters are not given, default values are used.
        let minimizer = NelderMeadBuilder::default()
            .xtol(1e-6f64)
            .ftol(1e-6f64)
            .maxiter(50000)
            .build()
            .unwrap();

        // Set the starting guess
        let args = Array::from_vec(vec![3.0, -8.3]);

        // Run the optimization
        let ans = minimizer.minimize(&function, args.view());

        // Print the optimized values
        println!("Final optimized arguments: {}", ans);
    }

    pub fn nsga(&self) {}

    pub fn bayes_search(&self, nb_iterations: usize) {
        // Objektivna funkcija.
        let f = |x: &[f64]| {
            let now = Instant::now();
            let timeframe = MIN_TIMEFRAME + TIMEFRAME_STEP * x[0] as u32;
            let len = x[1] as usize;
            let hline = x[2] as u32 as f32;
            let lline = x[3] as u32 as f32;
            let account = self.backtest(timeframe, len, hline, lline);
            unsafe {
                ELAPSED_NS += now.elapsed().as_nanos() as u64;
            }
            account.balance
        };
        // Definiranje područja pretraživanja.
        let input_interval = vec![
            (0., TIMEFRAME_RANGE as f64),
            (2., MAX_LEN as f64),
            (0., MAX_HLINE as f64),
            (0., MAX_LLINE as f64),
        ];
        // Broj iteracija optimizatora.
        // let nb_iterations = 100000;

        // Pokretanje optimizatora.
        Optimizer::new(&f, &input_interval, false)
            // "Optimizer" implementira "Iterator" "trait" te se preko njega pokreće.
            // Evaluiranje "nb_iterations" uzoraka.
            .skip(nb_iterations)
            // Traži se zadnja vrijednost kako bi se iterator izvršio.
            .next()
            .unwrap();
    }

    pub fn sharded_genetic_shearch(&mut self, n_shards: usize) {
        // Naznačuje se da se genetski algoritam mora podijeliti na 'n_shards' dijelova.
        self.n_shards = n_shards;
        let mut top = vec![Pair::default(); 10];
        for shard_i in 0..n_shards {
            self.shard_i = shard_i;
            // Izvršavanje genetske optimizacije nad manjem području vremenskog okvira.
            self.genetic_search();
            // Spremaju se najbolji parametri.
            let mut guard = self.top.lock();
            guard.sort_by_key(|x| OrderedFloat(x.balance));
            top.sort_by_key(|x| OrderedFloat(x.balance));
            let a: Vec<_> = guard.iter().map(|x| x.balance).collect();
            for ga_pair in 0..guard.len() {
                for pair in 0..top.len() {
                    if guard[ga_pair].balance > top[pair].balance {
                        top[pair] = guard[ga_pair].clone();
                        break;
                    }
                }
            }
            for result in guard.iter_mut() {
                *result = Pair::default();
            }
        }
        top.sort_by_key(|x| OrderedFloat(x.balance));
        self.top = Mutex::new(top);
        self.n_shards = 0;
    }

    pub fn hybrid_search(&mut self, bayes_iter: usize, pop: usize, gen: usize) {
        // Naznačuje se da se radi o hibridnom načinu rada genetskog algoritma.
        self.hybrid = true;
        let top = Mutex::new(vec![Pair::default(); 1]);
        // Funkcija koja se izvršava za svaku točku Bayesove optimizacije.
        let f = |x: &[f64]| {
            let timeframe = MIN_TIMEFRAME + TIMEFRAME_STEP * x[0] as u32;
            // Sprema se trenutni vremenski okvir.
            self.timeframe.store(timeframe, Ordering::SeqCst);
            // Izvšava se genetska optimizacija bez vremenskog okvira.
            self.genetic_search_param(pop, gen, 1.);
            // Spremaju se najbolje vrijednosti.
            let mut guard = self.top.lock();
            let ga_pair = guard
                .iter()
                .max_by_key(|x| OrderedFloat(x.balance))
                .unwrap();
            let mut top = top.lock();

            let (min_i, pair) = top
                .iter()
                .enumerate()
                .min_by_key(|x| OrderedFloat(x.1.balance))
                .unwrap();
            if ga_pair.balance > pair.balance {
                top[min_i] = ga_pair.clone();
            }
            let balance = ga_pair.balance;
            for result in guard.iter_mut() {
                *result = Pair::default();
            }
            balance
        };
        let input_interval = vec![(0., TIMEFRAME_RANGE as f64)];
        // let nb_iterations = 30;

        // Kreiranje novog Simple(x) optimizatora i evauliranje 'nb_iterations' broj točaka.
        let (max_value, coordinates) = Optimizer::new(&f, &input_interval, false)
            .map(|x| {
                // print!("hybrid {nb_iterations} ");
                x
            })
            .skip(bayes_iter)
            .next()
            .unwrap();
        self.top = top;
        self.hybrid = false;
    }

    pub fn summary(&self) {
        let compute = self.micros_compute.load(Ordering::SeqCst) as f32 / 1e6;
        let cache = self.micros_cache.load(Ordering::SeqCst) as f32 / 1e6;
        let timeframe = self.micros_timeframe.load(Ordering::SeqCst) as f32 / 1e6;
        println!(
            //             "compute_cost: {},
            // cache_cost: {},
            // timeframe_cost: {},
            "elapsed_compute_s: {},
        elapsed_cache_s: {},
        elapsed_timeframe_s: {},
        elapsed_total_s: {},",
            //             self.compute_cost.load(Ordering::SeqCst),
            //             self.cache_cost.load(Ordering::SeqCst),
            //             self.timeframe_cost.load(Ordering::SeqCst),
            compute,
            cache,
            timeframe,
            compute + cache + timeframe
        );
        unsafe {
            println!("\ncompute: {}s", ELAPSED_NS as f32 / 1e9);
        }
        self.top
            .lock()
            .iter()
            .sorted_by_key(|x| OrderedFloat(x.balance))
            .for_each(|x| println!("{:?}: {}", x.params, x.balance))
        // println!(
        //     "Top balance: {:?}",
        //         // .last()
        //         // .unwrap()
        //         // .balance
        // );
    }

    fn rank(&self, timeframe: u32, len: u32, hline: u32, lline: u32, account: &Account) {
        {
            let mut hist = self.histogram.lock();
            let i = (account.balance / self.bucket_size) as usize;
            (0..(i + 1).saturating_sub(hist.len())).for_each(|_| hist.push(0));
            hist[i] += 1;
        }
        let mut balance = account.balance;
        if account.n_trades < 30 {
            balance = 0.;
        }
        let mut top = self.top.lock();
        let (min_i, pair) = top
            .iter()
            .enumerate()
            .min_by_key(|x| OrderedFloat(x.1.balance))
            .unwrap();
        // if account.n_trades > 30 {
        //     println!("{:?}: {}, {}", [timeframe, len, hline, lline], account.n_trades, account.balance);
        // }
        if account.balance > pair.balance {
            let (max_i, max_pair) = top
                .iter()
                .enumerate()
                .max_by_key(|x| OrderedFloat(x.1.balance))
                .unwrap();
            if account.balance > max_pair.balance {
                // println!("{:?}: {}, {}", [timeframe, len, hline, lline], account.n_trades, account.balance);
                // print!("{:.2} ", account.balance);
                use std::io::Write;
                std::io::stdout().flush();
            }
            top[min_i] = Pair {
                params: [timeframe, len, hline, lline],
                balance: account.balance,
                account: account.clone(),
            };
        }
    }
}

// Implementacija osobine koja je potrebna za izvršavanje genetskog algoritma.
impl ObjectiveFunction for &Context {
    fn evaluate(&mut self, x: &DVector<f64>) -> f64 {
        // Pozivanje parallelne implementacije.
        ParallelObjectiveFunction::evaluate_parallel(self, x)
    }
}

// Biblioteka koristi dvije osobine, jednu za izvršavanje na jednoj jezgri procesora i drugu na više.
// Struktura implementira obije zbog lakšeg razvoja programa. Da bi se algoritam izvodio paralelno
// potrebno je pozvati "run_parallel" metodu.
impl ParallelObjectiveFunction for &Context {
    fn evaluate_parallel(&self, orig: &DVector<f64>) -> f64 {
        let now = Instant::now();
        let mut offset = 0;
        // U slučaju hibridnog načina rada se ignorira vremenski period.
        if self.hybrid {
            offset = 1;
        }
        for x in orig {
            if *x > 1. || *x < 0. {
                return 0.;
            }
        }
        // Orig je vektor brojeva između 0.0 i 1.0 on se skalira da bi stao u definirano područje
        // pretraživanja.
        let x = convert(orig, self.n_shards, self.shard_i);

        // Učitava se potrebni vremenski okvir u sljučaju GA + BO algoritma.
        let timeframe = if self.hybrid {
            self.timeframe.load(Ordering::SeqCst)
        } else {
            x[0].0 as u32
        };
        let period = x[1 - offset].0 as usize;
        let hline = x[2 - offset].0 as u32 as f32;
        let lline = x[3 - offset].0 as u32 as f32;
        let account = self.backtest(timeframe, period, hline, lline);
        self.time_taken_ns
            .fetch_add(now.elapsed().as_nanos() as u64, Ordering::SeqCst);
        account.balance as f64
    }
}

pub fn convert(x: &DVector<f64>, n_shards: usize, shard_i: usize) -> Vec<OrderedFloat<f64>> {
    let mut out = vec![OrderedFloat(0.); x.len()];
    let mut offset = 1;

    if x.len() == 4 {
        if n_shards == 1 {
            out[0] = OrderedFloat(
                MIN_TIMEFRAME as f64
                    + x[0].clamp(0., 1.) * TIMEFRAME_STEP as f64 * (TIMEFRAME_RANGE - 1) as f64,
            );
            out[0] = OrderedFloat((out[0].0 as u32 - out[0].0 as u32 % TIMEFRAME_STEP) as f64);
        } else {
            out[0] = OrderedFloat(
                MIN_TIMEFRAME as f64
                    + TIMEFRAME_STEP as f64 * TIMEFRAME_RANGE as f64 / n_shards as f64
                        * shard_i as f64
                    + x[0].clamp(0., 1.)
                        * TIMEFRAME_STEP as f64
                        * (TIMEFRAME_RANGE as usize / n_shards) as f64,
            );
            out[0] = OrderedFloat((out[0].0 as u32 - out[0].0 as u32 % TIMEFRAME_STEP) as f64);
        }
        offset = 0;
    }
    out[1 - offset] = OrderedFloat(2. + x[1 - offset].clamp(0., 1.) * (MAX_LEN - 2) as f64);
    out[2 - offset] = OrderedFloat(x[2 - offset].clamp(0., 1.) * MAX_HLINE as f64);
    out[3 - offset] = OrderedFloat(x[3 - offset].clamp(0., 1.) * MAX_LLINE as f64);

    out
}

#[derive(Default, Debug, Clone)]
pub struct Pair {
    pub params: [u32; 4],
    pub balance: f32,
    pub account: Account,
}
