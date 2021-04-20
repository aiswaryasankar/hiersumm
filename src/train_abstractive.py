#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import pickle
import signal
import time

import sentencepiece

from abstractive.model_builder import Summarizer
from abstractive.trainer_builder import build_trainer
from abstractive.predictor_builder import build_predictor
from abstractive.data_loader import load_dataset, load_dataset_live
import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from abstractive import data_loader, model_builder
from others import distributed
from others.logging import init_logger, logger

model_flags = [ 'emb_size', 'enc_hidden_size', 'dec_hidden_size', 'enc_layers', 'dec_layers', 'block_size', 'heads', 'ff_size', 'hier',
               'inter_layers', 'inter_heads', 'block_size']




def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def multi_main(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i

        procs.append(mp.Process(target=run, args=(args,
            device_id, error_queue), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()

def main(args):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
    init_logger(args.log_file)
    if (args.mode == 'train'):
        train(args, device_id)
    elif (args.mode == 'test'):
        # step = int(args.test_from.split('.')[-2].split('_')[-1])
        # validate(args, device_id, args.test_from, step)
        step = 1
        test(args, args.test_from, step)
    elif (args.mode == 'validate'):
        wait_and_validate(args, device_id)
    elif (args.mode == 'baseline'):
        baseline()
    elif (args.mode == 'print_flags'):
        print_flags()
    elif (args.mode == 'stats'):
        stats()


def train(args,device_id):
    init_logger(args.log_file)
    logger.info(str(args))

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])

    else:
        checkpoint = None

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<PAD>')
    symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'), 'PAD': word_padding_idx,
               'EOT': spm.PieceToId('<T>'), 'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>')}
    print(symbols)

    vocab_size = len(spm)

    def train_iter_fct():
        return data_loader.AbstractiveDataloader(args, load_dataset(args, 'train', shuffle=True), symbols, args.batch_size, device,
                                                 shuffle=True, is_test=False)

    model = Summarizer(args, word_padding_idx, vocab_size, device, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint)
    logger.info(model)
    trainer = build_trainer(args, device_id, model, symbols, vocab_size, optim)

    trainer.train(train_iter_fct, args.train_steps)


def wait_and_validate(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        ppl_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            ppl = validate(device_id, cp, step)
            ppl_lst.append((ppl, cp))
            max_step = ppl_lst.index(min(ppl_lst))
            if (i - max_step > 5):
                break
        ppl_lst = sorted(ppl_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(ppl_lst))
        for pp, cp in ppl_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test(args, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args,device_id, cp, step)
                    test(args,cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)

    #
    # while not os.path.exists(file_path):
    #     time.sleep(1)
    #
    # if os.path.isfile(file_path):
    # # read file
    # else:
    #     raise ValueError("%s isn't a file!" % file_path)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from

    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)


    opt = vars(checkpoint['opt'])

    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<PAD>')
    symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'), 'PAD': word_padding_idx,
               'EOT': spm.PieceToId('<T>'), 'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>')}

    vocab_size = len(spm)

    model = Summarizer(args, word_padding_idx, vocab_size, device, checkpoint)
    model.eval()

    valid_iter = data_loader.AbstractiveDataloader(args, load_dataset(args, 'valid', shuffle=False), symbols,
                                                   args.batch_size, device, shuffle=False, is_test=False)

    trainer = build_trainer(args, device_id, model, symbols, vocab_size, None)
    stats = trainer.validate(valid_iter)
    trainer._report_step(0, step, valid_stats=stats)
    return stats.ppl()


def test(args, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])

    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<PAD>')
    symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'), 'PAD': word_padding_idx,
               'EOT': spm.PieceToId('<T>'), 'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>')}

    vocab_size = len(spm)
    vocab = spm
    model = Summarizer(args, word_padding_idx, vocab_size, device, checkpoint)
    model.eval()

    data = ["President-elect Joe Biden criticized the Trump administration Tuesday for the pace of distributing COVID-19 vaccines and predicted that “things will get worse before they get better” when it comes to the pandemic. Biden encouraged Americans to “steel our spines” for challenges to come and predicted that “things are going to get worse before they get better.” Earlier this month, Trump administration officials said they planned to have 20 million doses of the vaccine distributed by the end of the year. At the current pace, Biden said, “it’s gonna take years, not months, to vaccinate the American people.", "The Second Stimulus Answers to Your Questions About the Stimulus Bill Updated Dec 30, 2020 The economic relief package will issue payments of $600 and distribute a federal unemployment benefit of $300 for at least 10 weeks. To receive assistance, households will have to meet several conditions: Household income (for 2020) cannot exceed more than 80 percent of the area median income; at least one household member must be at risk of homelessness or housing instability; and individuals must qualify for unemployment benefits or have experienced financial hardship — directly or indirectly — because of the pandemic. Robert Van Sant’s unemployment benefits of $484 a week don’t cover his monthly expenses of $2,200 in rent, utilities, internet access, food and other necessities.", "The majority leader, who effectively ruled out consideration of the House version stimulus check bill, has sought to couple the stimulus boost with unrelated provisions aimed at placating Trump’s demands to address legal protections for tech companies and his unsubstantiated claims of voter fraud in the 2020 election. McConnell said the House bill on the stimulus checks “has no realistic path to quickly pass the Senate,” even as Trump continues to harangue McConnell and GOP leaders over their refusal to go along with his demands. Sen. John Cornyn (R-Texas) predicted the Senate would follow the House and override Trump’s veto of the defense bill, calling it a matter of “how long people want to extend this out.", "The Second Stimulus Answers to Your Questions About the Stimulus Bill Updated Dec 30, 2020 The economic relief package will issue payments of $600 and distribute a federal unemployment benefit of $300 for at least 10 weeks. To receive assistance, households will have to meet several conditions: Household income (for 2020) cannot exceed more than 80 percent of the area median income; at least one household member must be at risk of homelessness or housing instability; and individuals must qualify for unemployment benefits or have experienced financial hardship — directly or indirectly — because of the pandemic. Robert Van Sant’s unemployment benefits of $484 a week don’t cover his monthly expenses of $2,200 in rent, utilities, internet access, food and other necessities.", "House Armed Services Chair Adam Smith (D-Wash.) told reporters last week that lawmakers’ “only option” if an override fails will be to attempt to pass the same defense agreement after President-elect Joe Biden takes office. Trump has also pushed for lawmakers to use the defense bill to repeal legal protections for social media companies, known as Section 230, but was rebuffed as Republicans and Democrats alike said it fell outside of the Armed Services Committee’s jurisdiction. Both the House and Senate passed a compromise defense bill last week with more than enough votes to overcome a veto, including strong support from Republicans.", "Top officials, including Speaker Nancy Pelosi and Mnuchin, are privately discussing contingency plans such as a stopgap spending bill if Trump does formally veto the measure by Monday, when funding is set to lapse. But it’s not clear how long that stopgap measure would last — or whether Trump would sign it, if it makes none of the changes he’s demanded, such as cuts to foreign aid, according to people familiar with the discussions. “Today, on Christmas Eve morning, House Republicans cruelly deprived the American people of the $2,000 that the President agreed to support,” Pelosi said in a statement.", "McConnell has also expressed concern that the vote could hurt GOP senators facing tough general election fights by alienating moderate voters. According to multiple people familiar with the discussion, the Senate GOP leader also asked Hawley several times to walk through how his objection would play out. The Missouri senator has focused his objections on Pennsylvania, arguing that it and other states failed to adhere to their own election laws. A Toomey spokesperson confirmed the account, saying: “Sen. Toomey made his views on Senator Hawley’s planned objection clear. Hawley instead sent an email to Senate Republicans after the call wrapped.", "President Trump spurned Democrats and Republicans alike on Wednesday as he left Washington defiantly for the holidays, vetoing a major defense bill, imperiling a COVID-19 relief package and setting the stage for a possible government shutdown after Christmas. Mr. Trump vetoed the $740 billion National Defense Authorization Act as promised, citing among his objections that lawmakers didn’t honor his late demand to repeal legal liability protections for big social media companies such as Twitter, Google and Facebook. Senate Armed Services Committee Chairman James Inhofe, Oklahoma Republican, said the president’s valid concerns about Big Tech companies shouldn’t derail the annual defense bill that sets national security priorities."]

    dataset_list = []
    mds = []

    for article in data:
        encoded = spm.Encode(article)
        mds.append(encoded)

    dataset_list = [{"src": mds, "tgt": [], "tgt_str": []}]
    # df = pd.DataFrame(dataset_list)
    # dataset = MultiNewsDataset(df)
    # dataloader = DataLoader(dataset, batch_size=1)

    # load_dataset(args, args.dataset, shuffle=False)
    test_iter = data_loader.AbstractiveDataloader(args, load_dataset_live(dataset_list), symbols,
                                                  args.valid_batch_size, device, shuffle=False, is_test=True)

    # test_iter = data_loader.AbstractiveDataloader(args, load_dataset(args, args.dataset, shuffle=False), symbols,
    #                                               args.valid_batch_size, device, shuffle=False, is_test=True)
    predictor = build_predictor(args, vocab, symbols, model, logger=logger)
    predictor.translate(test_iter, step, spm)

    # trainer.train(train_iter_fct, valid_iter_fct, FLAGS.train_steps, FLAGS.valid_steps)


def print_flags(args):
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    print(checkpoint['opt'])



def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' %gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        train(args,device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_file', default='', type=str)
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-visible_gpus', default='-0', type=str)
    parser.add_argument('-data_path', default='../../data/ranked_abs3_fast_b40/WIKI', type=str)
    parser.add_argument('-model_path', default='../../models', type=str)
    parser.add_argument('-vocab_path', default='../../data/spm9998_3.model', type=str)
    parser.add_argument('-train_from', default='', type=str)

    parser.add_argument('-trunc_src_ntoken', default=500, type=int)
    parser.add_argument('-trunc_tgt_ntoken', default=200, type=int)

    parser.add_argument('-emb_size', default=256, type=int)
    parser.add_argument('-enc_layers', default=8, type=int)
    parser.add_argument('-dec_layers', default=1, type=int)
    parser.add_argument('-enc_dropout', default=.2, type=float)
    parser.add_argument('-dec_dropout', default=0, type=float)
    parser.add_argument('-enc_hidden_size', default=256, type=int)
    parser.add_argument('-dec_hidden_size', default=256, type=int)
    parser.add_argument('-heads', default=8, type=int)
    parser.add_argument('-ff_size', default=1024, type=int)
    parser.add_argument("-hier", type=str2bool, nargs='?',const=True,default=True)


    parser.add_argument('-batch_size', default=10000, type=int)
    parser.add_argument('-valid_batch_size', default=10000, type=int)
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-lr', default=3, type=float)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-seed', default=0, type=int)

    parser.add_argument('-train_steps', default=20, type=int)
    parser.add_argument('-save_checkpoint_steps', default=20, type=int)
    parser.add_argument('-report_every', default=100, type=int)


    # multi-gpu
    parser.add_argument('-accum_count', default=1, type=int)
    parser.add_argument('-world_size', default=1, type=int)
    parser.add_argument('-gpu_ranks', default='0', type=str)

    # don't need to change flags
    parser.add_argument("-share_embeddings", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-share_decoder_embeddings", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument('-max_generator_batches', default=32, type=int)

    # flags for  testing
    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('-test_from', default='../../results', type=str)
    parser.add_argument('-result_path', default='../../results', type=str)
    parser.add_argument('-alpha', default=0, type=float)
    parser.add_argument('-length_penalty', default='wu', type=str)
    parser.add_argument('-beam_size', default=5, type=int)
    parser.add_argument('-n_best', default=1, type=int)
    parser.add_argument('-max_length', default=250, type=int)
    parser.add_argument('-min_length', default=20, type=int)
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-dataset', default='', type=str)
    parser.add_argument('-max_wiki', default=5, type=int)

    # flags for  hier
    # flags.DEFINE_boolean('old_inter_att', False, 'old_inter_att')
    parser.add_argument('-inter_layers', default='0', type=str)

    parser.add_argument('-inter_heads', default=8, type=int)
    parser.add_argument('-trunc_src_nblock', default=24, type=int)

    # flags for  graph


    # flags for  learning
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.998, type=float)
    parser.add_argument('-warmup_steps', default=8000, type=int)
    parser.add_argument('-decay_method', default='noam', type=str)
    parser.add_argument('-label_smoothing', default=0.1, type=float)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    args.inter_layers = [int(i) for i in args.inter_layers.split(',')]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    if(args.world_size>1):
        multi_main(args)
    else:
        main(args)

