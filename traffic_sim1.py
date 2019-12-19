# Traffic Light Simulator
# traffic_sim1.py
# Byron Burks

import asyncio
import random

async def red_light(duration):
    await asyncio.sleep(duration)
    return True


async def green_light(duration):
    await asyncio.sleep(duration)
    return True


async def amber_light(duration):
    await asyncio.sleep(duration)
    return True


async def street_seq(street_name, durations):
    """ durations = [green, amber, red, overlap] """
    print('Both streets have RED lights!')
    await red_light(durations[3])
    print('{street} has a GREEN light!'.format(street=street_name))
    await green_light(durations[0])
    print('{street} has an Amber light!'.format(street=street_name))
    await amber_light(durations[1])
    print('{street} has a RED light!'.format(street=street_name))
    return True


async def street_parallel1(street_name, semaphores, requests, delayQ, 
                           timeoutQ, advance1, quit_app, start=0, triggered=False):
    """ semaphores = [green, red1, red2] """
    amber_duration = 2
    overlap1 = 2
    trigger_delay = 3
    loop = asyncio.get_running_loop()
    while 1:
        async with semaphores[0]:
            # wait for green (value=1)
            await asyncio.sleep(overlap1)
            print(f'{street_name} has a GREEN light at {int(loop.time()-start)}!')
            await requests[0].wait()
            requests[0].clear()
            request_delay = await delayQ.get()
            await asyncio.sleep(request_delay)
            print(f'{street_name} has an AMBER light at {int(loop.time()-start)}!')
            await asyncio.sleep(amber_duration)
        print(f'{street_name} has a RED light at {int(loop.time()-start)}!')
        await advance1.wait()
        if quit_app.is_set():
            break
        else:
            advance1.clear()
        async with semaphores[1]:
            # change to red
            if triggered:
                await requests[1].wait()
                requests[1].clear()
                requests[0].set()
                delayQ.put_nowait(trigger_delay)
            else:
                timeout = await timeoutQ.get()
                await asyncio.sleep(timeout)
                requests[0].set()
                delayQ.put_nowait(0)
        await advance1.wait()
        if quit_app.is_set():
            break
        else:
            advance1.clear()
    return True


async def street_parallel2(street_name, semaphores, requests, delayQ, 
                           timeoutQ, advance2, quit_app, start=0, triggered=False):
    """ semaphores = [green, red1, red2] """
    amber_duration = 2
    overlap1 = 2
    trigger_delay = 3
    loop = asyncio.get_running_loop()
    print(f'{street_name} has a RED light at {int(loop.time()-start)}!')
    while 1:
        async with semaphores[2]:
            # change to red
            if triggered:
                await requests[1].wait()
                requests[1].clear()
                requests[0].set()
                delayQ.put_nowait(trigger_delay)
            else:
                timeout = await timeoutQ.get()
                await asyncio.sleep(timeout)
                requests[0].set()
                delayQ.put_nowait(0)
        await advance2.wait()
        if quit_app.is_set():
            break
        else:
            advance2.clear()
        async with semaphores[0]:
            # wait for green (value=1)
            await asyncio.sleep(overlap1)
            print(f'{street_name} has a GREEN light at {int(loop.time()-start)}!')
            await requests[0].wait()
            requests[0].clear()
            request_delay = await delayQ.get()
            await asyncio.sleep(request_delay)
            print(f'{street_name} has an AMBER light at {int(loop.time()-start)}!')
            await asyncio.sleep(amber_duration)
        print(f'{street_name} has a RED light at {int(loop.time()-start)}!')
        await advance2.wait()
        if quit_app.is_set():
            break
        else:
            advance2.clear()
    return True


async def trigger_handles1(street_name, semaphores, requests, 
                           timeoutQ, advance1, advance2, quit_app):
    """ semaphores = [green, red1, red2] """
    green_timeout = 6
    while 1:
        async with semaphores[1]:
            # insures that one street has a green light
            delay1 = random.randint(3, 10)
            await asyncio.sleep(delay1)
            requests[1].set()
            timeoutQ.put_nowait(green_timeout)
        async with semaphores[0]:
            # insures that a light change has occurred
            advance1.set()
            advance2.set()
            if quit_app.is_set():
                break
            else:
                pass
    return True


async def trigger_handles2(street_name, semaphores, requests, 
                           timeoutQ, advance1, advance2, quit_app):
    """ semaphores = [green, red1, red2] """
    green_timeout = 8
    while 1:
        async with semaphores[2]:
            # insures that the street has a red light
            delay1 = random.randint(3, 10)
            await asyncio.sleep(delay1)
            requests[1].set()
            timeoutQ.put_nowait(green_timeout)
        async with semaphores[0]:
            # insures that a light change has occurred
            advance1.set()
            advance2.set()
            if quit_app.is_set():
                break
            else:
                pass
    return True


async def simulation_timer(sim_duration, quit_app):
    """ stops the simulation after 'sim_duration' seconds have elapsed """
    # loop = asyncio.get_running_loop()
    await asyncio.sleep(sim_duration)
    quit_app.set()
    return True


async def intersection_version1():
    """ simplest version """
    green_duration = 5
    amber_duration = 1
    overlap1 = 1
    red_duration = green_duration + amber_duration
    durations = [green_duration, amber_duration, 
                 red_duration, overlap1]
    for _ in range(2):
        await street_seq('Smith St.', durations)
        await street_seq('Brown St.', durations)
    return True


async def intersection_version2():
    """ with handoffs and triggers """
    semaphores = [asyncio.Semaphore(1), asyncio.Semaphore(1), asyncio.Semaphore(1)]
    requests = [asyncio.Event(), asyncio.Event()]
    delayQ = asyncio.Queue()
    timeoutQ = asyncio.Queue()
    advance1 = asyncio.Event()
    advance2 = asyncio.Event()
    quit_app = asyncio.Event()
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    await asyncio.gather(
        trigger_handles2('Brown St.', semaphores, requests, timeoutQ, advance1, advance2, quit_app), 
        street_parallel1('Smith St.', semaphores, requests, delayQ, timeoutQ, advance1, quit_app, start=start_time), 
        street_parallel2('Brown St.', semaphores, requests, delayQ, timeoutQ, advance2, quit_app, start=start_time, triggered=True), 
        simulation_timer(40, quit_app), 
    )
    return True


if __name__ == "__main__":
    asyncio.run(intersection_version2())





